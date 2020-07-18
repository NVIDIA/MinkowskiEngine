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
#include <vector>
#include <sstream>

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
  if (!v.empty()) {
    auto actual_delim = ", ";
    auto delim = "";
    out << '[';
    for (const auto &elem : v) {
      out << delim << elem;
      delim = actual_delim;
    }
    out << "]\n";
  }
  return out;
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
      formatter << " assertion (" #condition << ") faild. ";                   \
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

#endif // UTILS
