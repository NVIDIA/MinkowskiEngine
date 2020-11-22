/*
 * Copyright (c) 2020 NVIDIA Corporation.
 * Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
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
#ifndef MATH_FUNCTIONS_HPP
#define MATH_FUNCTIONS_HPP

#include "mkl_alternate.hpp"

namespace minkowski {

template <typename Dtype>
void cpu_gemm(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE TransA,
              const CBLAS_TRANSPOSE TransB, const int M, const int N,
              const int K, const Dtype alpha, const Dtype *A, const Dtype *B,
              const Dtype beta, Dtype *C);

template <typename Dtype>
void cpu_add(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void cpu_mul(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void cpu_div(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void cpu_axpy(const int N, const Dtype alpha, const Dtype *X, Dtype *Y);

} // end namespace minkowski

#endif // MATH_FUNCTIONS
