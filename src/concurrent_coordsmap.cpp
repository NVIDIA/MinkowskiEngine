/* Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 * Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 * of the code.
 */
#include "concurrent_coordsmap.hpp"

void ConcurrentCoordsMap::set_threads(int num_threads_) {
  if (num_threads_ > 0) {
    num_threads = num_threads_;
    tbb::task_scheduler_init init(num_threads);
    // tbb::global_control c(tbb::global_control::max_allowed_parallelism,
    // num_threads);
  }
}

ConcurrentCoordsMap::ConcurrentCoordsMap(int ncols_,
                                         const set<int> &batch_indices)
    : ncols(ncols_), nrows(batch_indices.size()) {

  coords.resize(ncols * nrows);
  fill(coords.begin(), coords.end(), 0);
  int *p_coords = coords.data();
  int c = 0;

  Coord<int> coord; // only a wrapper.
  coord.size = ncols;

  for (int b : batch_indices) {
    // Create a key
#ifdef BATCH_FIRST
    p_coords[0] = b;
#else
    p_coords[ncols - 1] = b;
#endif
    coord.ptr = p_coords;

    // Add to the map
    map[coord] = c;

    // Move the heads
    p_coords += ncols;
    c++;
  }
}

// Initializations
//
// Preferably, use std::move to coords_.
// returns mapping: out_coord row index to in_coord row index
pair<vector<int>, set<int>>
ConcurrentCoordsMap::initialize(vector<int> &&coords_, int nrows_, int ncols_,
                                bool force_remap) {
  // Copy to the local vars
  nrows = nrows_;
  ncols = ncols_;
  coords = coords_;

  // tbb::tick_count t0 = tbb::tick_count::now();
  //
  // Use the input order to define the map coords -> index
  tbb::concurrent_unordered_map<int, int> concurr_batch_indices;
  tbb::parallel_for(tbb::blocked_range<int>(0, nrows),
                    [&](const tbb::blocked_range<int> &r) {
                      Coord<int> coord; // only a wrapper.
                      coord.size = ncols;
                      for (int i = r.begin(); i != r.end(); ++i) {
                        coord.ptr = &coords[ncols * i];

                        // Only the first pointer among all coords with the
                        // same coord will be used.
                        map[coord] = i;

      // batch index and set an arbitrary value
#ifdef BATCH_FIRST
                        concurr_batch_indices[*coord.ptr] = 0;
#else
                        concurr_batch_indices[*(coord.ptr + ncols - 1)] = 0;
#endif
                      }
                    },
                    tbb::auto_partitioner());

  // tbb::tick_count t1 = tbb::tick_count::now();
  set<int> batch_indices;
  for (const auto &i : concurr_batch_indices)
    batch_indices.insert(i.first);

  // cout << "Creation: " << (t1 - t0).seconds() << endl;
  int unique_size = map.size();

  // If the mapping size is different, remap the entire coordinates
  if (force_remap && (nrows != unique_size)) {
    // Assign a unique index to an item.
    //
    // Then assign the unique index to original row index mapping. Order does
    // not matter.  This randomized order (through multi-threads) will be the
    // new unique index.
    tbb::concurrent_vector<int> mapping;
    mapping.reserve(map.size());
    // tbb::tick_count t2 = tbb::tick_count::now();
    tbb::parallel_for(map.range(),
                      [&](decltype(map)::const_range_type &r) {
                        for (const auto &i : r) {
                          mapping.push_back(i.second);
                        }
                      },
                      tbb::auto_partitioner());

    // Assign the new unique (randomized) permutation to remap the map.
    tbb::parallel_for(tbb::blocked_range<int>(0, map.size()),
                      [&](const tbb::blocked_range<int> &r) {
                        Coord<int> coord; // only a wrapper.
                        coord.size = ncols;
                        for (int i = r.begin(); i != r.end(); ++i) {
                          coord.ptr = &coords[ncols * mapping[i]];
                          map[coord] = i;
                        }
                      },
                      tbb::auto_partitioner());

    nrows = unique_size;
    updateUniqueCoords(coords, mapping);

    // Copy the concurrent vector to std vector
    auto std_mapping = vector<int>(mapping.size());
    tbb::parallel_for(tbb::blocked_range<int>(0, mapping.size()),
                      [&](const tbb::blocked_range<int> &r) {
                        for (int i = r.begin(); i != r.end(); ++i) {
                          std_mapping[i] = mapping[i];
                        }
                      },
                      tbb::auto_partitioner());
    // tbb::tick_count t3 = tbb::tick_count::now();
    // cout << "Remapping: " << (t3 - t2).seconds() << endl;

    ASSERT(nrows == map.size(), "Map size mismatch", nrows, "!=", map.size());
    ASSERT(coords.size() / ncols == nrows, "Map size mismatch");

    return make_pair(move(std_mapping), move(batch_indices));
  } else {
    // Return null vector
    return make_pair(vector<int>(0), move(batch_indices));
  }
}

// Generate strided version of the input coordinate map.
// returns mapping: out_coord row index to in_coord row index
ConcurrentCoordsMap
ConcurrentCoordsMap::stride(const vector<int> &tensor_strides) {
  ASSERT(tensor_strides.size() == ncols - 1, "Invalid tensor strides");

  // Copy
  vector<int> tmp_strided_coords(coords);
  // Map for unique coords
  ConcurrentCoordsInnerMap tmp_strided_map;

  // Strided coords
  tbb::parallel_for(tbb::blocked_range<int>(0, nrows),
                    [&](const tbb::blocked_range<int> r) {
                      Coord<int> coord;
                      coord.size = ncols;
                      for (int i = r.begin(); i != r.end(); ++i) {
                        coord.ptr = &tmp_strided_coords[i * ncols];
                        stride_ptr<int>(coord.ptr, tensor_strides);
                        tmp_strided_map[coord] = i;
                      }
                    },
                    tbb::auto_partitioner());

  // Create a new concurrent map
  ConcurrentCoordsMap strided_coords_map;

  strided_coords_map.nrows = tmp_strided_map.size();
  strided_coords_map.ncols = ncols;

  auto &strided_map = strided_coords_map.map;
  auto &strided_coords = strided_coords_map.coords;
  strided_coords.resize(tmp_strided_map.size() * ncols);

  // Define mapping from the old row index (tmp_strided_map k-v value) to new
  tbb::concurrent_vector<int> mapping;
  mapping.reserve(tmp_strided_map.size());
  tbb::parallel_for(tmp_strided_map.range(),
                    [&](decltype(tmp_strided_map)::const_range_type &r) {
                      for (const auto &i : r) {
                        mapping.push_back(i.second);
                      }
                    },
                    tbb::auto_partitioner());

  // Assign the new unique (randomized) permutation to remap the map.
  tbb::parallel_for(tbb::blocked_range<int>(0, tmp_strided_map.size()),
                    [&](const tbb::blocked_range<int> &r) {
                      Coord<int> coord; // only a wrapper.
                      coord.size = ncols;
                      for (int i = r.begin(); i != r.end(); ++i) {
                        coord.ptr = &strided_coords[ncols * i];
                        copy_n(&tmp_strided_coords[ncols * mapping[i]], ncols,
                               coord.ptr);
                        strided_map[coord] = i;
                      }
                    },
                    tbb::auto_partitioner());

  ASSERT(strided_coords_map.size() == strided_coords_map.nrows,
         "Map size mismatch");

  return strided_coords_map;
}

ConcurrentCoordsMap ConcurrentCoordsMap::stride_region(const Region &region) {
  ASSERT(region.tensor_strides.size() == ncols - 1, "Invalid tensor strides");

  ConcurrentCoordsInnerMap tmp_map;

  // Assign a unique index to an item.
  //
  // Then assign the unique index to original row index mapping. Order does
  // not matter.  This randomized order (through multi-threads) will be the
  // new unique index.
  // tbb::tick_count t2 = tbb::tick_count::now();
  tbb::parallel_for(this->map.range(),
                    [&](decltype(map)::const_range_type &r) {
                      Region cregion(region);
                      Coord<int> coord;
                      coord.size = ncols;
                      for (const auto &i : r) {
                        cregion.set_bounds(i.first.ptr);
                        for (const auto &point : cregion) {
                          int *p_coord =
                              tbb::scalable_allocator<int>().allocate(ncols);
                          copy_n(point.begin(), ncols, p_coord);
                          coord.ptr = p_coord;
                          tmp_map[coord] = 0; // set random value
                        }
                      }
                    },
                    tbb::auto_partitioner());

  // p_mapping
  tbb::concurrent_vector<int *> p_mapping;
  p_mapping.reserve(tmp_map.size());

  // Assign the new unique (randomized) permutation to remap
  tbb::parallel_for(tmp_map.range(),
                    [&](decltype(tmp_map)::const_range_type &r) {
                      for (const auto &i : r) {
                        p_mapping.push_back(i.first.ptr);
                      }
                    },
                    tbb::auto_partitioner());

  // Create a new concurrent map
  ConcurrentCoordsMap out_coords_map;

  // Use the index to copy the coords to the out_coords
  vector<int> &out_coords = out_coords_map.coords;
  auto &out_inner_map = out_coords_map.map;
  out_coords_map.ncols = ncols;
  out_coords_map.nrows = tmp_map.size();
  out_coords.resize(tmp_map.size() * ncols);

  tbb::parallel_for(tbb::blocked_range<int>(0, p_mapping.size()),
                    [&](const tbb::blocked_range<int> &r) {
                      Coord<int> coord;
                      coord.size = ncols;
                      for (int i = r.begin(); i != r.end(); ++i) {
                        // Set the key
                        coord.ptr = &out_coords[i * ncols];
                        copy_n(p_mapping[i], ncols, coord.ptr);
                        tbb::scalable_allocator<int>().deallocate(p_mapping[i],
                                                                  ncols);
                        // Set the mapping
                        out_inner_map[coord] = i;
                      }
                    },
                    tbb::auto_partitioner());

  ASSERT(out_coords_map.size() == out_coords_map.nrows, "Map size mismatch");

  return out_coords_map;
}

ConcurrentCoordsMap ConcurrentCoordsMap::prune(bool *p_keep, int n) {
  ASSERT(nrows == n,
         "The number of elements in the map mismatch the keep vector");

  tbb::concurrent_vector<int> keep_inds;

  tbb::parallel_for(tbb::blocked_range<int>(0, n),
                    [&](const tbb::blocked_range<int> &r) {
                      for (int i = r.begin(); i != r.end(); ++i) {
                        if (p_keep[i] > 0) {
                          keep_inds.push_back(i);
                        }
                      }
                    },
                    tbb::auto_partitioner());

  int n_keep = keep_inds.size();
  ASSERT(n_keep > 0,
         "The number of elements to keep is not a positive number: ", n_keep);

  ConcurrentCoordsMap out_coords_map;
  out_coords_map.ncols = ncols;
  out_coords_map.nrows = n_keep;

  vector<int> &out_coords = out_coords_map.coords;
  out_coords.resize(n_keep * ncols);
  auto &out_inner_map = out_coords_map.map;

  tbb::parallel_for(tbb::blocked_range<int>(0, n_keep),
                    [&](const tbb::blocked_range<int> &r) {
                      Coord<int> coord;
                      coord.size = ncols;
                      for (int i = r.begin(); i != r.end(); ++i) {
                        // i is the out_coords row index
                        // keep_inds[i] is the coords to copy from
                        copy_n(&coords[ncols * keep_inds[i]], ncols,
                               &out_coords[i * ncols]);
                        coord.ptr = &out_coords[i * ncols];
                        out_inner_map[coord] = i;
                      }
                    },
                    tbb::auto_partitioner());

  ASSERT(out_coords_map.size() == out_coords_map.nrows, "Map size mismatch");

  return out_coords_map;
}

// Get the unique coords from the input coordinates and mapping
void ConcurrentCoordsMap::updateUniqueCoords(
    vector<int> &coords_, const tbb::concurrent_vector<int> &mapping) {

  // New map and the corresponding new coords
  ConcurrentCoordsInnerMap new_map;
  vector<int> new_coords(map.size() * ncols);

  tbb::parallel_for(map.range(), [&](decltype(map)::const_range_type &r) {
    Coord<int> coord;
    coord.size = ncols;
    for (const auto &i : r) {
      // Set the new coord
      coord.ptr = &new_coords[i.second * ncols];
      copy_n(&coords_[ncols * mapping[i.second]], ncols, coord.ptr);
      // Insert the new coord
      new_map[coord] = i.second; // replace with the new pointer
    }
  });

  ASSERT(map.size() == new_map.size(), "Remapping sizes different.")

  // Must move to replace the map explicitly.
  map = move(new_map);
  coords = move(new_coords);
}

// Generate in-out kernel maps
InOutMapsPair<int>
ConcurrentCoordsMap::kernel_map(const ConcurrentCoordsMap &out_coords_map,
                                const Region &region) const {
  int K = region.size();
  vector<tbb::concurrent_vector<int *>> in_out_pairs(K);
  for (auto &in_out_pair : in_out_pairs)
    in_out_pair.reserve(out_coords_map.size());

  // Most time consuming part
  tbb::parallel_for(out_coords_map.map.range(),
                    [&](decltype(out_coords_map.map)::const_range_type &r) {
                      Region cregion(region);
                      vector<int> vec_coord(ncols);
                      Coord<int> coord(vec_coord.data(), ncols);
                      int kernel_ind;

                      // For all range
                      for (const auto &o : r) {
                        // set the bounds for the current region
                        cregion.set_bounds(o.first.ptr);

                        // For elements in the current region
                        kernel_ind = 0;
                        for (const auto &point : cregion) {
                          copy_n(point.begin(), ncols, coord.ptr);

                          // If the input coord exists
                          const auto &iter_map = map.find(coord);
                          if (iter_map != map.end()) {
                            int *in_out_pair =
                                tbb::scalable_allocator<int>().allocate(2);
                            // In index
                            in_out_pair[0] = iter_map->second;
                            // Out index
                            in_out_pair[1] = o.second;
                            // Then save the pointer
                            in_out_pairs[kernel_ind].push_back(in_out_pair);
                          }
                          // Post processings
                          kernel_ind++;
                        }
                      }
                    },
                    tbb::auto_partitioner());

  // After we find all in out pairs, copy them to vectors
  //
  // This is pretty fast
  InOutMaps<int> in_maps;
  InOutMaps<int> out_maps;
  for (int k = 0; k < K; k++) {
    int n = in_out_pairs[k].size();
    vector<int> in_indices(n), out_indices(n);
    tbb::parallel_for(tbb::blocked_range<int>(0, n),
                      [&](const tbb::blocked_range<int> &r) {
                        for (int i = r.begin(); i != r.end(); ++i) {
                          int *pair = in_out_pairs[k][i];
                          in_indices[i] = pair[0];
                          out_indices[i] = pair[1];
                          // Deallocate when done copying the values
                          tbb::scalable_allocator<int>().deallocate(pair, 2);
                        }
                      },
                      tbb::auto_partitioner());
    in_maps.push_back(move(in_indices));
    out_maps.push_back(move(out_indices));
  }

  return make_pair(in_maps, out_maps);
}

// Generate in-out kernel maps
InOutMapsPair<int> ConcurrentCoordsMap::pruned_kernel_map(
    const ConcurrentCoordsMap &out_coords_map) const {
  tbb::concurrent_vector<int *> in_out_pairs;

  tbb::parallel_for(
      out_coords_map.map.range(),
      [&](decltype(out_coords_map.map)::const_range_type &r) {
        Coord<int> coord;
        coord.size = ncols;

        // For all range
        for (const auto &o : r) {
          coord.ptr = o.first.ptr;

          // If the input coord exists
          const auto &iter_map = map.find(coord);
          ASSERT(iter_map != map.end(),
                 "Key not found: ", PtrToString(coord.ptr, ncols));
          int *in_out_pair = tbb::scalable_allocator<int>().allocate(2);
          // In index
          in_out_pair[0] = iter_map->second;
          // Out index
          in_out_pair[1] = o.second;
          // Then save the pointer
          in_out_pairs.push_back(in_out_pair);
        }
      },
      tbb::auto_partitioner());

  // After we find all in out pairs, copy them to vectors
  //
  // This is pretty fast
  InOutMaps<int> in_maps;
  InOutMaps<int> out_maps;
  int n = in_out_pairs.size();
  vector<int> in_indices(n), out_indices(n);
  tbb::parallel_for(tbb::blocked_range<int>(0, n),
                    [&](const tbb::blocked_range<int> &r) {
                      for (int i = r.begin(); i != r.end(); ++i) {
                        int *pair = in_out_pairs[i];
                        in_indices[i] = pair[0];
                        out_indices[i] = pair[1];
                        // Deallocate when done copying the values
                        tbb::scalable_allocator<int>().deallocate(pair, 2);
                      }
                    },
                    tbb::auto_partitioner());
  in_maps.push_back(move(in_indices));
  out_maps.push_back(move(out_indices));

  return make_pair(in_maps, out_maps);
}

// Generate in-out kernel maps
InOutMapsPair<int> ConcurrentCoordsMap::global_reduction_map(
    const ConcurrentCoordsMap &gout_coords_map) const {
  tbb::concurrent_vector<int *> in_out_pairs;

  // Most time consuming part
  tbb::parallel_for(
      map.range(),
      [&](decltype(map)::const_range_type &r) {
        vector<int> coord_vec(ncols);
        fill(coord_vec.begin(), coord_vec.end(), 0);

        Coord<int> coord(coord_vec.data(), ncols);

        // For all range
        for (const auto &i : r) {
#ifdef BATCH_FIRST
          coord_vec[0] = i.first[0];
#else
          coord_vec[ncols - 1] = i.first[ncols - 1];
#endif

          // If the input coord exists
          const auto &iter_gmap = gout_coords_map.find(coord);
          ASSERT(iter_gmap != gout_coords_map.end(), "Key,",
                 ArrToString(coord_vec), ", not found in the global map",
                 to_string(i.first.ptr - coords.data()), ", ",
                 PtrToString(&coords[i.second * ncols], ncols));

          int *in_out_pair = tbb::scalable_allocator<int>().allocate(2);

          // In index
          in_out_pair[0] = i.second;
          // Out index
          in_out_pair[1] = iter_gmap->second;
          // Then save the pointer
          in_out_pairs.push_back(in_out_pair);
        }
      },
      tbb::auto_partitioner());

  // After we find all in out pairs, copy them to vectors
  //
  // This is pretty fast
  InOutMaps<int> in_maps;
  InOutMaps<int> out_maps;

  int n = in_out_pairs.size();
  vector<int> in_indices(n), out_indices(n);
  tbb::parallel_for(tbb::blocked_range<int>(0, n),
                    [&](const tbb::blocked_range<int> &r) {
                      for (int i = r.begin(); i != r.end(); ++i) {
                        int *pair = in_out_pairs[i];
                        in_indices[i] = pair[0];
                        out_indices[i] = pair[1];
                        // Deallocate when done copying the values
                        tbb::scalable_allocator<int>().deallocate(pair, 2);
                      }
                    },
                    tbb::auto_partitioner());
  in_maps.push_back(move(in_indices));
  out_maps.push_back(move(out_indices));

  return make_pair(in_maps, out_maps);
}

// Generate in-out kernel maps
InOutMapsPair<int>
ConcurrentCoordsMap::stride_map(const ConcurrentCoordsMap &out_coords_map,
                                const vector<int> &tensor_strides) const {

  tbb::concurrent_vector<int *> in_out_pairs;

  // Strided coords
  tbb::parallel_for(
      map.range(),
      [&](decltype(map)::const_range_type &r) {
        vector<int> coord_vec(ncols);
        Coord<int> coord(coord_vec.data(), ncols);
        for (const auto &i : r) {
          stride_copy<int>(i.first.ptr, coord.ptr, tensor_strides);

          const auto &iter_omap = out_coords_map.find(coord);
          ASSERT(iter_omap != out_coords_map.end(),
                 "Key with : ", PtrToString(i.first.ptr, ncols), " -> ",
                 PtrToString(coord.ptr, ncols),
                 "doesn't exist in the out coords map");
          int *in_out_pair = tbb::scalable_allocator<int>().allocate(2);
          // In index
          in_out_pair[0] = i.second;
          // Out index
          in_out_pair[1] = iter_omap->second;
          // Then save the pointer
          in_out_pairs.push_back(in_out_pair);
        }
      },
      tbb::auto_partitioner());

  // After we find all in out pairs, copy them to vectors
  InOutMaps<int> in_maps;
  InOutMaps<int> out_maps;

  int n = in_out_pairs.size();
  vector<int> in_indices(n), out_indices(n);
  tbb::parallel_for(tbb::blocked_range<int>(0, n),
                    [&](const tbb::blocked_range<int> &r) {
                      for (int i = r.begin(); i != r.end(); ++i) {
                        int *pair = in_out_pairs[i];
                        in_indices[i] = pair[0];
                        out_indices[i] = pair[1];
                        // Deallocate when done copying the values
                        tbb::scalable_allocator<int>().deallocate(pair, 2);
                      }
                    },
                    tbb::auto_partitioner());

  in_maps.push_back(move(in_indices));
  out_maps.push_back(move(out_indices));

  return make_pair(in_maps, out_maps);
}

void ConcurrentCoordsMap::print() const {
  for (const auto &i : map) {
    cout << &coords[ncols * i.second] << " " << i.first.ptr << " ";
    for (int k = 0; k < i.first.size; k++)
      cout << i.first.ptr[k] << " ";
    cout << ":" << i.second << endl;
  }
}
