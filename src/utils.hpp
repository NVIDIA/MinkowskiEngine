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
#ifndef UTILS
#define UTILS
#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef CPU_ONLY
#include <thrust/host_vector.h>
#endif

namespace minkowski {

struct timer {

  void tic() { m_start = std::chrono::high_resolution_clock::now(); }

  double toc() {
    return std::chrono::duration<double>(
               std::chrono::high_resolution_clock::now() - m_start)
        .count();
  }

  std::chrono::high_resolution_clock::time_point m_start;
};

template <typename T>
std::ostream &print_vector(std::ostream &out, const T &v) {
  if (!v.empty()) {
    auto actual_delim = ", ";
    auto delim = "";
    out << '[';
    for (const auto &elem : v) {
      out << delim << elem;
      delim = actual_delim;
    }
    out << "]";
  }
  return out;
}

#ifndef CPU_ONLY
template <typename T>
std::ostream &operator<<(std::ostream &out, const thrust::host_vector<T> &v) {
  return print_vector(out, v);
}
#endif

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
  return print_vector(out, v);
}

template <typename T> std::string ArrToString(const T &arr) {
  std::string buf = "[";
  for (const auto &a : arr) {
    buf += std::to_string(a) + ", ";
  }
  buf += arr.empty() ? "]" : "\b\b]";
  return buf;
}

template <typename T> std::string PtrToString(const T *ptr, int size) {
  std::string buf = "[";
  for (int i = 0; i < size; i++) {
    buf += (i ? ", " : "") + std::to_string(ptr[i]);
  }
  buf += "]";
  return buf;
}

class Formatter {
public:
  Formatter() {}
  ~Formatter() {}

  template <typename Type> Formatter &operator<<(const Type &value) {
    stream_ << value;
    return *this;
  }

  template <typename Type> void append(Type value) { stream_ << value; }

  // Recursively append arguments
  template <typename Type, typename... Args>
  void append(Type value, Args... args) {
    stream_ << value << " ";
    append(args...);
  }

  std::string str() const { return stream_.str(); }
  operator std::string() const { return stream_.str(); }

  enum ConvertToString { to_str };

  std::string operator>>(ConvertToString) { return stream_.str(); }

private:
  std::stringstream stream_;
  Formatter(const Formatter &);
  Formatter &operator=(Formatter &);
};

#define OVERFLOW_IF(condition, ...)                                            \
  {                                                                            \
    if (condition) {                                                           \
      Formatter formatter;                                                     \
      formatter << __FILE__ << ":" << __LINE__ << ",";                         \
      formatter << " overflow condition (" #condition << "). ";                \
      formatter.append(__VA_ARGS__);                                           \
      throw std::overflow_error(formatter.str());                              \
    }                                                                          \
  }

#define ASSERT(condition, ...)                                                 \
  {                                                                            \
    if (!(condition)) {                                                        \
      Formatter formatter;                                                     \
      formatter << __FILE__ << ":" << __LINE__ << ",";                         \
      formatter << " assertion (" #condition << ") failed. ";                  \
      formatter.append(__VA_ARGS__);                                           \
      throw std::runtime_error(formatter.str());                               \
    }                                                                          \
  }

#define WARNING(condition, ...)                                                \
  {                                                                            \
    if (condition) {                                                           \
      Formatter formatter;                                                     \
      formatter << __FILE__ << ":" << __LINE__ << ",";                         \
      formatter << " (" #condition << ") ";                                    \
      formatter.append(__VA_ARGS__);                                           \
      std::cerr << formatter.str() << std::endl;                               \
    }                                                                          \
  }

#ifdef __CUDACC__
#define MINK_CUDA_HOST_DEVICE __host__ __device__
#define MINK_CUDA_DEVICE __device__
#else
#define MINK_CUDA_HOST_DEVICE
#define MINK_CUDA_DEVICE
#define THRUST_CHECK(condition) condition;
#endif

#define COLOR "\033[31;1m"
#define RESET "\033[0m"

#ifdef DEBUG
#define __DEBUG(...)                                                           \
  {                                                                            \
    Formatter formatter;                                                       \
    formatter << COLOR << __FILE__ << ":" << __LINE__ << RESET << " ";         \
    formatter.append(__VA_ARGS__);                                             \
    std::cerr << formatter.str() << "\n";                                      \
  }
// #define __DEBUG(msg, ...) fprintf(stderr, COLOR msg "%c" RESET, __VA_ARGS__);
// #define LOG_DEBUG(...) __DEBUG(__VA_ARGS__, '\n')
#define LOG_DEBUG(...) __DEBUG(__VA_ARGS__)
#else
#define LOG_DEBUG(...) (void)0
#endif

#define __WARN(...)                                                            \
  {                                                                            \
    Formatter formatter;                                                       \
    formatter << COLOR << "WARNING:" << __FILE__ << ":" << __LINE__ << RESET   \
              << " ";                                                          \
    formatter.append(__VA_ARGS__);                                             \
    std::cerr << formatter.str() << "\n";                                      \
  }

#define LOG_WARN(...) __WARN(__VA_ARGS__)

class simple_range {
  using index_type = uint32_t;

public:
  // member typedefs provided through inheriting from std::iterator
  class iterator
      : public std::iterator<std::input_iterator_tag, // iterator_category
                             index_type,              // value_type
                             index_type,              // difference_type
                             const index_type *,      // pointer
                             index_type               // reference
                             > {
    // custom iterator members
    index_type m_num;

  public:
    explicit iterator(index_type _num = 0) : m_num(_num) {}
    iterator &operator++() {
      m_num++;
      return *this;
    }
    iterator operator++(int) {
      iterator retval = *this;
      ++(*this);
      return retval;
    }
    int32_t operator-(iterator const &other) const {
      return m_num - other.m_num;
    }
    bool operator==(iterator const &other) const {
      return m_num == other.m_num;
    }
    bool operator!=(iterator const &other) const { return !(*this == other); }
    bool operator!=(iterator &&other) const { return !(*this == other); }
    reference operator*() const { return m_num; }
  };

  simple_range(index_type from, index_type to) : m_from(from), m_to(to) {
    ASSERT(m_to > m_from, "Invalid range");
  }
  simple_range(index_type to) : m_from(0), m_to(to) {
    ASSERT(m_to > m_from, "Invalid range");
  }
  simple_range(simple_range const &) = delete;
  simple_range(simple_range &&other) : m_from(other.m_from), m_to(other.m_to) {}

  std::string to_string() {
    Formatter formatter;
    formatter << "Range: " << m_from << " -- " << m_to << "\n";
    return formatter.str();
  }

  iterator begin() { return iterator(m_from); }
  iterator end() { return iterator(m_to); }

  index_type const m_from, m_to;
};

} // end namespace minkowski

#endif // UTILS
