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
#ifndef SMALL_VECTOR_HPP
#define SMALL_VECTOR_HPP

#include <algorithm>
#include <cstdlib> // std::malloc, std::realloc, std::free, std::size_t
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <utility>

#include "../utils.hpp"

namespace minkowski {

/// \brief A dynamic array type optimized for few elements.
///
template <typename T, std::size_t N = 4> class small_vector {
public:
  static_assert(std::is_trivially_copyable<T>::value, "");

  // Types for consistency with STL containers
  using value_type = T;
  using iterator = T *;
  using const_iterator = T const *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using size_type = std::size_t;

  /// The default-initialized array has zero size.
  small_vector() {}

  /// Like `std::vector`, you can initialize with a size and a default value.
  explicit small_vector(size_type sz, T newval = T()) { resize(sz, newval); }

  /// Like `std::vector`, you can initialize with a set of values in braces.
  small_vector(std::initializer_list<T> const init) {
    resize(init.size());
    std::copy(init.begin(), init.end(), data_);
  }

  /// Copy constructor, initializes with a copy of `other`.
  small_vector(small_vector const &other) {
    resize(other.size_);
    std::copy(other.data_, other.data_ + size_, data_);
  }

  /// \brief Cast constructor, initializes with a copy of `other`.
  /// Casting done as default in C++
  template <typename O> explicit small_vector(small_vector<O> const &other) {
    resize(other.size());
    std::transform(other.data(), other.data() + size_, data_,
                   [](O const &v) { return static_cast<value_type>(v); });
  }

  /// Move constructor, initializes by stealing the contents of `other`.
  small_vector(small_vector &&other) noexcept { steal_data_from(other); }

  // Destructor, no need for documentation
  ~small_vector() {
    free_array();
  }

  /// Copy assignment, copies over data from `other`.
  small_vector &operator=(small_vector const &other) {
    if (this != &other) {
      resize(other.size_);
      std::copy(other.data_, other.data_ + size_, data_);
    }
    return *this;
  }

  /// Move assignment, steals the contents of `other`.
  small_vector &operator=(small_vector &&other) noexcept {
    // Self-assignment is not valid for move assignment, not testing for it
    // here.
    free_array();
    steal_data_from(other);
    return *this;
  }

  /// Swaps the contents of two arrays.
  void swap(small_vector &other) {
    using std::swap;
    if (is_dynamic()) {
      if (other.is_dynamic()) {
        // both have dynamic memory
        swap(data_, other.data_);
      } else {
        // *this has dynamic memory, other doesn't
        other.data_ = data_;
        data_ = stat_;
        std::move(other.stat_, other.stat_ + other.size_, stat_);
      }
    } else {
      if (other.is_dynamic()) {
        // other has dynamic memory, *this doesn't
        data_ = other.data_;
        other.data_ = other.stat_;
        std::move(stat_, stat_ + size_, other.stat_);
      } else {
        // both have static memory
        std::swap_ranges(stat_, stat_ + std::max(size_, other.size_),
                         other.stat_);
      }
    }
    swap(size_, other.size_);
  }

  /// \brief Resizes the array, making it either larger or smaller. Initializes
  /// new elements with `newval`.
  void resize(size_type newsz, T newval = T()) {
    if (newsz == size_) {
      return;
    } // NOP
    if (newsz > static_size_) {
      if (is_dynamic()) {
        // expand or contract heap data
        T *tmp = static_cast<T *>(std::realloc(data_, newsz * sizeof(T)));
        // std::cout << "   small_vector realloc\n";
        if (tmp == nullptr) {
          throw std::bad_alloc();
        }
        data_ = tmp;
        if (newsz > size_) {
          std::fill(data_ + size_, data_ + newsz, newval);
        }
        size_ = newsz;
      } else {
        // move from static to heap data
        // We use malloc because we want to be able to use realloc; new cannot
        // do this.
        T *tmp = static_cast<T *>(std::malloc(newsz * sizeof(T)));
        // std::cout << "   small_vector malloc\n";
        if (tmp == nullptr) {
          throw std::bad_alloc();
        }
        std::move(stat_, stat_ + size_, tmp);
        data_ = tmp;
        std::fill(data_ + size_, data_ + newsz, newval);
        size_ = newsz;
      }
    } else {
      if (is_dynamic()) {
        // move from heap to static data
        if (newsz > 0) {
          std::move(data_, data_ + newsz, stat_);
        }
        free_array();
        size_ = newsz;
        data_ = stat_;
      } else {
        // expand or contract static data
        if (newsz > size_) {
          std::fill(stat_ + size_, stat_ + newsz, newval);
        }
        size_ = newsz;
      }
    }
  }

  /// Clears the contents of the array, set its length to 0.
  void clear() { resize(0); }

  /// Checks whether the array is empty (size is 0).
  bool empty() const { return size_ == 0; }

  /// Returns the size of the array.
  size_type size() const { return size_; }

  /// Accesses an element of the array
  T &operator[](size_type index) { return *(data_ + index); }
  /// Accesses an element of the array
  T const &operator[](size_type index) const { return *(data_ + index); }

  /// Accesses the first element of the array
  T &front() { return *data_; }
  /// Accesses the first element of the array
  T const &front() const { return *data_; }

  /// Accesses the last element of the array
  T &back() { return *(data_ + size_ - 1); }
  /// Accesses the last element of the array
  T const &back() const { return *(data_ + size_ - 1); }

  /// Returns a pointer to the underlying data
  T *data() { return data_; };
  /// Returns a pointer to the underlying data
  T const *data() const { return data_; };

  /// Returns an iterator to the beginning
  iterator begin() { return data_; }
  /// Returns an iterator to the beginning
  const_iterator begin() const { return data_; }
  /// Returns an iterator to the end
  iterator end() { return data_ + size_; }
  /// Returns an iterator to the end
  const_iterator end() const { return data_ + size_; }
  /// Returns a reverse iterator to the beginning
  reverse_iterator rbegin() { return reverse_iterator(end()); }
  /// Returns a reverse iterator to the beginning
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  /// Returns a reverse iterator to the end
  reverse_iterator rend() { return reverse_iterator(begin()); }
  /// Returns a reverse iterator to the end
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  /// \brief Adds a value at the given location, moving the current value at
  /// that location and subsequent values forward by one.
  void insert(size_type index, T const &value) {
    ASSERT(index <= size_, "");
    resize(size_ + 1);
    if (index < size_ - 1) {
      std::move_backward(data_ + index, data_ + size_ - 1, data_ + size_);
    }
    *(data_ + index) = value;
  }

  /// Adds a value to the back.
  void push_back(T const &value) {
    resize(size_ + 1);
    back() = value;
  }

  /// Adds all values in source array to the back.
  void push_back(small_vector const &values) {
    size_type index = size_;
    resize(size_ + values.size_);
    for (size_type ii = 0; ii < values.size_; ++ii) {
      data_[index + ii] = values.data_[ii];
    }
  }

  /// Removes the value at the given location, moving subsequent values forward
  /// by one.
  void erase(size_type index) {
    ASSERT(index < size_, "");
    if (index < size_ - 1) {
      std::move(data_ + index + 1, data_ + size_, data_ + index);
    }
    resize(size_ - 1);
  }

  /// Removes the value at the back.
  void pop_back() {
    ASSERT(size_ > 0, "");
    resize(size_ - 1);
  }

private:
  constexpr static size_type static_size_ = N;
  size_type size_ = 0;
  T stat_[static_size_]; // static data
  T *data_ = stat_; // dynamic data
  // The alternate implementation, where data_ and stat_ are in a union
  // to reduce the amount of memory used, requires a test for every data
  // access. Data access is most frequent, it's worth using a little bit
  // more memory to avoid that test.

  bool is_dynamic() noexcept { return data_ != stat_; }

  void free_array() noexcept {
    if (is_dynamic()) {
      std::free(data_);
    }
  }

  void steal_data_from(small_vector &other) noexcept {
    if (other.is_dynamic()) {
      size_ = other.size_;
      data_ = other.data_;       // move pointer
      other.size_ = 0;           // so other won't deallocate the memory space
      other.data_ = other.stat_; // make sure other is consistent
    } else {
      size_ = other.size_;
      data_ = stat_;
      std::move(other.data_, other.data_ + size_, data_);
    }
  }
};

//
// Other operators and convenience functions
//

/// \brief Writes the array to a stream
template <typename T>
inline std::ostream &operator<<(std::ostream &os,
                                small_vector<T> const &array) {
  os << "{";
  auto it = array.begin();
  if (it != array.end()) {
    os << *it;
    while (++it != array.end()) {
      os << ", " << *it;
    };
  }
  os << "}";
  return os;
}

template <typename T>
inline void swap(small_vector<T> &v1, small_vector<T> &v2) {
  v1.swap(v2);
}

} // namespace minkowski

#endif // SMALL_VECTOR_HPP
