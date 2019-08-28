/*  Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of
 *  this software and associated documentation files (the "Software"), to deal in
 *  the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 *  of the Software, and to permit persons to whom the Software is furnished to do
 *  so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 *  Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 *  Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 *  of the code.
 */
#ifndef INSTANTIATION
#define INSTANTIATION

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname)                                           \
  char gInstantiationGuard##classname;                                         \
  template class classname<float>;                                             \
  // template class classname<double>
  //
#define INSTANTIATE_CLASS_DIM(CLASSNAME)                                       \
  template class CLASSNAME<1>;                                                 \
  template class CLASSNAME<2>;                                                 \
  template class CLASSNAME<3>;                                                 \
  template class CLASSNAME<4>;                                                 \
  template class CLASSNAME<5>;                                                 \
  template class CLASSNAME<6>;                                                 \
  template class CLASSNAME<7>;                                                 \
  template class CLASSNAME<8>;                                                 \
  template class CLASSNAME<9>;

#define INSTANTIATE_CLASS_DIM_ITYPE(CLASSNAME, ITYPE)                          \
  template class CLASSNAME<1, ITYPE>;                                          \
  template class CLASSNAME<2, ITYPE>;                                          \
  template class CLASSNAME<3, ITYPE>;                                          \
  template class CLASSNAME<4, ITYPE>;                                          \
  template class CLASSNAME<5, ITYPE>;                                          \
  template class CLASSNAME<6, ITYPE>;                                          \
  template class CLASSNAME<7, ITYPE>;                                          \
  template class CLASSNAME<8, ITYPE>;                                          \
  template class CLASSNAME<9, ITYPE>;

#define SWITCH_DIM_TYPES(func, Dtype, Itype, ...)                              \
  switch (D) {                                                                 \
  case 1:                                                                      \
    func<1, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 2:                                                                      \
    func<2, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 3:                                                                      \
    func<3, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 4:                                                                      \
    func<4, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 5:                                                                      \
    func<5, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 6:                                                                      \
    func<6, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 7:                                                                      \
    func<7, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 8:                                                                      \
    func<8, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 9:                                                                      \
    func<9, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  default:                                                                     \
    throw std::invalid_argument(Formatter() << "Not supported D " << D);       \
  }

#endif
