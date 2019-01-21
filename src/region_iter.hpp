#ifndef REGION
#define REGION

#include "common.hpp"

template <uint8_t D, typename Itype> class RegionIterator;
template <uint8_t D, typename Itype> class Region {
public:
  Region(const Coord<D, Itype> &center_, const Arr<D, int> &pixel_dists,
         const Arr<D, int> &kernel_size, const Arr<D, int> &dilations,
         int region_type, const Itype *p_offset, int n_offset);

  Region(const Coord<D, Itype> &lower_bound_, const Arr<D, int> &pixel_dists,
         const Arr<D, int> &kernel_size, const Arr<D, int> &dilations,
         int region_type, const Itype *p_offset, int n_offset,
         bool use_lower_bound);

  RegionIterator<D, Itype> begin() { return RegionIterator<D, Itype>(*this); }
  RegionIterator<D, Itype> end() { return RegionIterator<D, Itype>(*this); }

  int region_type;
  Arr<D, Itype> pixel_dists, kernel_size, dilations;
  const Itype *p_offset, n_offset;
  Coord<D, Itype> center;
  Coord<D, Itype> lb;
  Coord<D, Itype> ub;
  bool use_lower_bound;
};

template <uint8_t D, typename Itype> class RegionIterator {
private:
  int curr_axis, offset_ind;
  const Region<D, Itype> &region;
  Coord<D, Itype> point;

public:
  bool done;
  RegionIterator(const Region<D, Itype> &region)
      : curr_axis(0), offset_ind(0), region(region), done(false) {
    // First point
    switch (region.region_type) {
    case 0:
      point = region.lb;
      break;
    case 1:
      // First, start from the origin
      point = region.center;
      break;
    case 2:
      // First offset
      for (int i = 0; i < D; i++) {
        point[i] = region.center[i] + region.p_offset[i];
      }
      point[D] = region.center[D];
      break;
    }
  }
  RegionIterator<D, Itype> &operator++() {
    switch (region.region_type) {
    case 0:
      // Iterate only from 0 to D-1, point[D] reserved for batch index
      for (int d = 0; d < D;) {
        point[d] += region.dilations[d] *
                    region.pixel_dists[d]; // point is initialized as lb
        if (point[d] <= region.ub[d])
          break;
        point[d] = region.lb[d];
        d++;
        if (d >= D) {
          done = true; // Signal to operator!= to end iteration
          break;
        }
      }
      return *this;
    case 1:
      while (curr_axis < D) {
        // Go through [4, 5, 1, 2] when kernel_size = 5, and ceter = 3.
        // Center passed at the initialization
        point[curr_axis] +=
            region.dilations[curr_axis] * region.pixel_dists[curr_axis];
        // skip if the current point is crossing the center
        if (point[curr_axis] == region.center[curr_axis]) {
          curr_axis++;
          continue;
        }
        // When it passes the last point, reset to the lower bound
        if (point[curr_axis] > region.ub[curr_axis])
          point[curr_axis] = region.lb[curr_axis];
        break;
      }
      if (curr_axis >= D) { // if it has past all axes
        done = true;
      }
      return *this;
    case 2:         // custom offset
      offset_ind++; // already past the first offset
      if (offset_ind >= region.n_offset) {
        done = true;
      } else {
        for (int i = 0; i < D; i++) {
          point[i] = region.center[i] + region.p_offset[D * offset_ind + i];
        }
      }
      return *this;
    }
    // To make the compiler happy
    return *this;
  }
  Coord<D, Itype> &operator*() { return point; }
};

// Only to be used for checking the end point of range based for loops.
template <uint8_t D, typename Itype>
inline bool operator!=(const RegionIterator<D, Itype> &lhs,
                       const RegionIterator<D, Itype> &rhs) {
  return !lhs.done;
}
#endif
