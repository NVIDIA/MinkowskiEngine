/*
 * Copyright (c) 2020 NVIDIA Corporation.
 * Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 * Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 * of the code.
 */
#include "gpu.cuh"
#include "math_functions.cuh"

#include <cusparse.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>
#include <torch/script.h>

namespace minkowski {

#define BLOCK_SIZE 128

template <typename Itype, typename Dtype>
__global__ void
unique_row2num_nonzero(const int n, Dtype *__restrict__ d_num_nonzero,
                       const Itype *__restrict__ unique_row_ptr,
                       const Dtype *__restrict__ reduced_val_ptr) {
  CUDA_KERNEL_LOOP(index, n) {
    d_num_nonzero[unique_row_ptr[index]] = reduced_val_ptr[index];
  }
}

template <typename Itype, typename Dtype>
__global__ void inverse_val(const int n, Dtype *__restrict__ d_sorted_val,
                            const Itype *__restrict__ sorted_row,
                            const Dtype *__restrict__ reduced_val) {
  auto const tx = threadIdx.x;
  auto const bx = blockIdx.x;
  auto const x = blockDim.x * bx + tx;

  if (x < n) {
    Itype row = sorted_row[x];
    d_sorted_val[x] = 1.0 / __ldg(&reduced_val[row]);
  }
}

cudaDataType getTensorCudaDataType(torch::Tensor const &self) {
  cudaDataType cuda_data_type;
  switch (self.scalar_type()) {
  case torch::ScalarType::Float:
    cuda_data_type = CUDA_R_32F;
    break;
  case torch::ScalarType::Double:
    cuda_data_type = CUDA_R_64F;
    break;
  default:
    TORCH_CHECK(false, "Tensor types must be either float32 or float64");
    break;
  }
  return cuda_data_type;
}

// template <typename th_int_type>
// torch::Tensor sort_row_col(torch::Tensor &rows, torch::Tensor &cols) {
//   AT_DISPATCH_FLOATING_TYPES(vals.scalar_type(), "coo_spmm", [&] {
// 		  }
// }

template <typename th_int_type>
torch::Tensor coo_spmm(torch::Tensor const &rows, torch::Tensor const &cols,
                       torch::Tensor const &vals, int64_t const dim_i,
                       int64_t const dim_j, torch::Tensor const &mat2,
                       int64_t const spmm_algorithm_id, bool const is_sorted) {
#if defined __HIP_PLATFORM_HCC__
  TORCH_CHECK(false, "spmm sparse-dense is not supported on HIP");
#elif defined(_WIN32) || defined(_WIN64)
  TORCH_CHECK(false, "spmm sparse-dense CUDA is not supported on Windows");
#elif !defined(CUDART_VERSION)
  TORCH_CHECK(false, "CUDART_VERSION not defined");
#endif

  constexpr bool is_int32 = std::is_same<th_int_type, int32_t>::value;
  constexpr bool is_int64 = std::is_same<th_int_type, int64_t>::value;

  cusparseSpMMAlg_t mm_alg;
#if defined(CUDART_VERSION) && (CUDART_VERSION < 10010)
  TORCH_CHECK(false, "spmm sparse-dense requires CUDA 10.1 or greater");
#elif defined(CUDART_VERSION) && (CUDART_VERSION >= 10010) &&                  \
    (CUDART_VERSION < 11000)
  switch (spmm_algorithm_id) {
  case 1:
    mm_alg = CUSPARSE_COOMM_ALG1;
    break;
  case 2:
    mm_alg = CUSPARSE_COOMM_ALG2;
    break;
  case 3:
    mm_alg = CUSPARSE_COOMM_ALG3;
    break;
  default:
    TORCH_CHECK(false, "Invalid algorithm id.", spmm_algorithm_id);
    mm_alg = CUSPARSE_MM_ALG_DEFAULT;
  }
  TORCH_CHECK(is_int32, "int64 cusparseSpMM requires CUDA 11.1 or greater");
#elif defined(CUDART_VERSION) && (CUDART_VERSION >= 11000)
  switch (spmm_algorithm_id) {
  case 1:
    mm_alg = CUSPARSE_SPMM_COO_ALG1;
    break;
  case 2:
    mm_alg = CUSPARSE_SPMM_COO_ALG2;
    break;
  case 3:
    mm_alg = CUSPARSE_SPMM_COO_ALG3;
    break;
  case 4:
    mm_alg = CUSPARSE_SPMM_COO_ALG4;
    break;
  default:
    TORCH_CHECK(false, "Invalid algorithm id.", spmm_algorithm_id);
    mm_alg = CUSPARSE_SPMM_ALG_DEFAULT;
  }
  TORCH_CHECK(is_int32, "int64 coosort not implemented");
  // coosort not supported with int64 || (is_int64 && (mm_alg ==
  // CUSPARSE_SPMM_COO_ALG4)));
#endif

  at::ScalarType int_scalar_type = std::is_same<th_int_type, int32_t>::value
                                       ? at::ScalarType::Int
                                       : at::ScalarType::Long;

  ASSERT(rows.scalar_type() == int_scalar_type, "int type mismatch.");

  ASSERT(rows.scalar_type() == cols.scalar_type(),
         "rows and cols must have the same scalar type.");
  ASSERT(rows.scalar_type() == cols.scalar_type(),
         "rows and cols must have the same scalar type.");
  ASSERT(vals.scalar_type() == mat2.scalar_type(),
         "vals and mat2 must have the same scalar type.");

  ASSERT(rows.is_contiguous(), "rows must be contiguous");
  ASSERT(cols.is_contiguous(), "cols must be contiguous");
  ASSERT(vals.is_contiguous(), "vals must be contiguous");

  ASSERT(rows.is_cuda(), "rows must be CUDA, but got CPU");
  ASSERT(cols.is_cuda(), "cols must be CUDA, but got CPU");
  ASSERT(vals.is_cuda(), "vals must be CUDA, but got CPU");
  ASSERT(mat2.is_cuda(), "mat2 must be CUDA, but got CPU");
  ASSERT(at::cuda::check_device({rows, cols, vals, mat2}),
         "All inputs must be on the same device.");

  ASSERT(mat2.dim() == 2, "Tensor 'mat2' must have 2 dims, but has ",
         mat2.dim());

  // int64_t dim_i = self.size(0);
  // int64_t dim_j = self.size(1);
  int64_t dim_k = mat2.size(1);

  // Create tensors to view just the current set of matrices
  int64_t const nnz = rows.numel();

  LOG_DEBUG("COO_SPMM with dim_i:", dim_i, ", dim_j:", dim_j,
            ", mat2.size(0):", mat2.size(0), ", mat2.size(1):", mat2.size(1),
            "nnz:", nnz);
  LOG_DEBUG("Creating a result mat shaped (", dim_k, ",", dim_i, ")");
  torch::Tensor result = at::zeros({dim_k, dim_i}, mat2.options());

  if ((dim_j == 0) || (dim_k == 0) || (nnz == 0)) {
    LOG_DEBUG("Skipping matmul dim_j:", dim_j, "dim_k:", dim_k, "nnz:", nnz);
    result.transpose_(0, 1);
    return result;
  }

  // Dense matrices have to be contiguous for cusparseSpMM to work
  torch::Tensor const mat2_contig = mat2.contiguous();
  // Issue 308
  // auto cusparse_handle = at::cuda::getCurrentCUDASparseHandle();
  auto stream = at::cuda::getCurrentCUDAStream();
  cusparseHandle_t cusparse_handle = getCurrentCUDASparseHandle();
  cusparseSetStream(cusparse_handle, stream);

  torch::Scalar beta = 0;
  torch::Scalar alpha = 1;

  cudaDataType cuda_data_type = getTensorCudaDataType(mat2_contig);
  th_int_type *row_indices_ptr =
      reinterpret_cast<th_int_type *>(rows.data_ptr());
  th_int_type *col_indices_ptr =
      reinterpret_cast<th_int_type *>(cols.data_ptr());

  // Iterate through each set of 2D matrices within the 3D
  // tensor inputs, performing a matrix multiply with each
  AT_DISPATCH_FLOATING_TYPES(vals.scalar_type(), "coo_spmm", [&] {
    scalar_t alpha_val = alpha.to<scalar_t>();
    scalar_t beta_val = beta.to<scalar_t>();

    scalar_t *values_ptr = reinterpret_cast<scalar_t *>(vals.data_ptr());
    scalar_t *mat2_ptr = reinterpret_cast<scalar_t *>(mat2_contig.data_ptr());
    scalar_t *result_ptr = reinterpret_cast<scalar_t *>(result.data_ptr());

    th_int_type *sorted_row_ptr, *sorted_col_ptr;
    scalar_t *sorted_val_ptr;
    //////////////////////////////////////
    // Sort the sparse matrix COO
    LOG_DEBUG("Is sorted", is_sorted);
    if (!is_sorted) {
      sorted_row_ptr =
          (th_int_type *)c10::cuda::CUDACachingAllocator::raw_alloc(
              2 * nnz * sizeof(th_int_type));
      sorted_col_ptr = sorted_row_ptr + nnz;
      sorted_val_ptr = (scalar_t *)c10::cuda::CUDACachingAllocator::raw_alloc(
          nnz * sizeof(scalar_t));
      LOG_DEBUG("Allocated sorted row col val", nnz);

      // Copy the indices
      CUDA_CHECK(cudaMemcpy(sorted_row_ptr, row_indices_ptr,
                            nnz * sizeof(th_int_type),
                            cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(sorted_col_ptr, col_indices_ptr,
                            nnz * sizeof(th_int_type),
                            cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(sorted_val_ptr, values_ptr, nnz * sizeof(scalar_t),
                            cudaMemcpyDeviceToDevice));

      THRUST_CHECK(thrust::sort_by_key(thrust::device,            //
                                       sorted_row_ptr,            // key begin
                                       sorted_row_ptr + nnz,      // key end
                                       thrust::make_zip_iterator( // value begin
                                           thrust::make_tuple(    //
                                               sorted_col_ptr,    //
                                               sorted_val_ptr     //
                                               )                  //
                                           )));
      LOG_DEBUG("sorted row", cudaDeviceSynchronize());
    } else {
      sorted_row_ptr = row_indices_ptr;
      sorted_col_ptr = col_indices_ptr;
      sorted_val_ptr = values_ptr;
      LOG_DEBUG("Initialized ptrs from inputs");
    }
    //////////////////////////////////////

    size_t workspace_buffer_size = 0;
    void *workspace_buffer = nullptr;

    cusparseSpMatDescr_t sparse_descr;
    CUSPARSE_CHECK(cusparseCreateCoo(
        &sparse_descr,     //
        dim_i, dim_j, nnz, //
        reinterpret_cast<void *>(sorted_row_ptr),
        reinterpret_cast<void *>(sorted_col_ptr),
        reinterpret_cast<void *>(sorted_val_ptr), //
        std::is_same<th_int_type, int32_t>::value ? CUSPARSE_INDEX_32I
                                                  : CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, cuda_data_type));

    cusparseDnMatDescr_t dense_descr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_descr,                       //
                                       dim_k, dim_j, dim_k,                //
                                       reinterpret_cast<void *>(mat2_ptr), //
                                       cuda_data_type, CUSPARSE_ORDER_COL));

    cusparseDnMatDescr_t result_descr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&result_descr,                        //
                                       dim_i, dim_k, dim_i,                  //
                                       reinterpret_cast<void *>(result_ptr), //
                                       cuda_data_type, CUSPARSE_ORDER_COL));
    LOG_DEBUG("initialized matrices", cudaGetLastError());

    size_t required_workspace_buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE, (void *)&alpha_val, sparse_descr,
        dense_descr, (void *)&beta_val, result_descr, cuda_data_type, mm_alg,
        &required_workspace_buffer_size));
    LOG_DEBUG("Buffer size:", required_workspace_buffer_size);

    if (required_workspace_buffer_size > workspace_buffer_size) {
      if (workspace_buffer != nullptr) {
        cudaFree(workspace_buffer);
      }
      workspace_buffer_size = required_workspace_buffer_size;
      LOG_DEBUG("cudaMallocManaged");
      cudaMallocManaged(&workspace_buffer, workspace_buffer_size);
    }
    CUSPARSE_CHECK(cusparseSpMM(cusparse_handle,                  //
                                CUSPARSE_OPERATION_NON_TRANSPOSE, //
                                CUSPARSE_OPERATION_TRANSPOSE,     //
                                (void *)&alpha_val,               //
                                sparse_descr, dense_descr,        //
                                (void *)&beta_val, result_descr,  //
                                cuda_data_type, mm_alg, workspace_buffer));

#ifdef DEBUG
    LOG_DEBUG("SPMM", cudaDeviceSynchronize());
    CUDA_CHECK_DEBUG(cudaDeviceSynchronize());
#endif

    // Cleanup
    CUSPARSE_CHECK(cusparseDestroySpMat(sparse_descr));
    CUSPARSE_CHECK(cusparseDestroyDnMat(dense_descr));
    CUSPARSE_CHECK(cusparseDestroyDnMat(result_descr));

    if (!is_sorted) {
      LOG_DEBUG("Dealloc");
      c10::cuda::CUDACachingAllocator::raw_delete((void *)sorted_row_ptr);
      c10::cuda::CUDACachingAllocator::raw_delete((void *)sorted_val_ptr);
    }

    if (workspace_buffer != nullptr) {
      cudaFree(workspace_buffer);
    }
    LOG_DEBUG("Dealloc finished", cudaDeviceSynchronize());
  });

  // Need to transpose the result matrices since cusparse stores
  // them in column-major order in memory
  result.transpose_(0, 1);

  CUDA_CHECK(cudaDeviceSynchronize());

  return result;
}

template torch::Tensor
coo_spmm<int32_t>(torch::Tensor const &rows, torch::Tensor const &cols,
                  torch::Tensor const &vals, int64_t const dim_i,
                  int64_t const dim_j, torch::Tensor const &mat2,
                  int64_t const spmm_algorithm_id, bool const is_sorted);

template <typename th_int_type>
std::vector<torch::Tensor> // output, sorted rows, sorted cols, sorted vals.
coo_spmm_average(torch::Tensor const &rows, torch::Tensor const &cols,
                 int64_t const dim_i, int64_t const dim_j,
                 torch::Tensor const &mat2, int64_t const spmm_algorithm_id) {
#if defined __HIP_PLATFORM_HCC__
  TORCH_CHECK(false, "spmm sparse-dense is not supported on HIP");
#elif defined(_WIN32) || defined(_WIN64)
  TORCH_CHECK(false, "spmm sparse-dense CUDA is not supported on Windows");
#elif !defined(CUDART_VERSION)
  TORCH_CHECK(false, "CUDART_VERSION not defined");
#endif

  constexpr bool is_int32 = std::is_same<th_int_type, int32_t>::value;
  constexpr bool is_int64 = std::is_same<th_int_type, int64_t>::value;

  cusparseSpMMAlg_t mm_alg;
#if defined(CUDART_VERSION) && (CUDART_VERSION < 10010)
  TORCH_CHECK(false, "spmm sparse-dense requires CUDA 10.1 or greater");
#elif defined(CUDART_VERSION) && (CUDART_VERSION >= 10010) &&                  \
    (CUDART_VERSION < 11000)
  switch (spmm_algorithm_id) {
  case 1:
    mm_alg = CUSPARSE_COOMM_ALG1;
    break;
  case 2:
    mm_alg = CUSPARSE_COOMM_ALG2;
    break;
  case 3:
    mm_alg = CUSPARSE_COOMM_ALG3;
    break;
  default:
    TORCH_CHECK(false, "Invalid algorithm id.", spmm_algorithm_id);
    mm_alg = CUSPARSE_MM_ALG_DEFAULT;
  }
  TORCH_CHECK(is_int32, "int64 cusparseSpMM requires CUDA 11.1 or greater");
#elif defined(CUDART_VERSION) && (CUDART_VERSION >= 11000)
  switch (spmm_algorithm_id) {
  case 1:
    mm_alg = CUSPARSE_SPMM_COO_ALG1;
    break;
  case 2:
    mm_alg = CUSPARSE_SPMM_COO_ALG2;
    break;
  case 3:
    mm_alg = CUSPARSE_SPMM_COO_ALG3;
    break;
  case 4:
    mm_alg = CUSPARSE_SPMM_COO_ALG4;
    break;
  default:
    TORCH_CHECK(false, "Invalid algorithm id.", spmm_algorithm_id);
    mm_alg = CUSPARSE_SPMM_ALG_DEFAULT;
  }
  TORCH_CHECK(is_int32, "int64 coosort not implemented");
  // coosort not supported with int64 || (is_int64 && (mm_alg ==
  // CUSPARSE_SPMM_COO_ALG4)));
#endif

  at::ScalarType int_scalar_type = std::is_same<th_int_type, int32_t>::value
                                       ? at::ScalarType::Int
                                       : at::ScalarType::Long;
  cudaDataType cuda_data_type = getTensorCudaDataType(mat2);

  ASSERT(rows.scalar_type() == int_scalar_type, "int type mismatch.");

  ASSERT(rows.scalar_type() == cols.scalar_type(),
         "rows and cols must have the same scalar type.");
  ASSERT(rows.scalar_type() == cols.scalar_type(),
         "rows and cols must have the same scalar type.");

  ASSERT(rows.is_contiguous(), "rows must be contiguous");
  ASSERT(cols.is_contiguous(), "cols must be contiguous");

  ASSERT(rows.is_cuda(), "rows must be CUDA, but got CPU");
  ASSERT(cols.is_cuda(), "cols must be CUDA, but got CPU");
  ASSERT(mat2.is_cuda(), "mat2 must be CUDA, but got CPU");
  ASSERT(at::cuda::check_device({rows, cols, mat2}),
         "All inputs must be on the same device.");

  ASSERT(mat2.dim() == 2, "Tensor 'mat2' must have 2 dims, but has ",
         mat2.dim());

  // int64_t dim_i = self.size(0);
  // int64_t dim_j = self.size(1);
  int64_t dim_k = mat2.size(1);

  // Create tensors to view just the current set of matrices
  int64_t const nnz = rows.numel();

  auto int_options =
      torch::TensorOptions({at::kCUDA, at::cuda::current_device()})
          .dtype(int_scalar_type)
          .requires_grad(false);

  torch::Tensor result = at::zeros({dim_k, dim_i}, mat2.options());
  torch::Tensor sorted_row_col = at::zeros({2, nnz}, int_options);
  torch::Tensor sorted_val = at::zeros({nnz}, mat2.options());

  if ((dim_j == 0) || (dim_k == 0) || (nnz == 0)) {
    LOG_DEBUG("Skipping matmul dim_j:", dim_j, "dim_k:", dim_k, "nnz:", nnz);
    result.transpose_(0, 1);
    return {result, sorted_row_col, sorted_val};
  }

  // Dense matrices have to be contiguous for cusparseSpMM to work
  torch::Tensor const mat2_contig = mat2.contiguous();
  // Issue 308
  // auto cusparse_handle = at::cuda::getCurrentCUDASparseHandle();
  auto stream = at::cuda::getCurrentCUDAStream();
  cusparseHandle_t cusparse_handle = getCurrentCUDASparseHandle();
  cusparseSetStream(cusparse_handle, stream);

  torch::Scalar beta = 0;
  torch::Scalar alpha = 1;

  th_int_type *row_indices_ptr =
      reinterpret_cast<th_int_type *>(rows.data_ptr());
  th_int_type *col_indices_ptr =
      reinterpret_cast<th_int_type *>(cols.data_ptr());

  // Iterate through each set of 2D matrices within the 3D
  // tensor inputs, performing a matrix multiply with each
  AT_DISPATCH_FLOATING_TYPES(mat2.scalar_type(), "coo_spmm", [&] {
    scalar_t alpha_val = alpha.to<scalar_t>();
    scalar_t beta_val = beta.to<scalar_t>();

    scalar_t *mat2_ptr = reinterpret_cast<scalar_t *>(mat2_contig.data_ptr());
    scalar_t *result_ptr = reinterpret_cast<scalar_t *>(result.data_ptr());

    //////////////////////////////////////
    // Sort the sparse matrix COO
    th_int_type *sorted_row_ptr =
        reinterpret_cast<th_int_type *>(sorted_row_col.data_ptr());
    th_int_type *sorted_col_ptr = sorted_row_ptr + nnz;

    // Copy the indices
    CUDA_CHECK(cudaMemcpy(sorted_row_ptr, row_indices_ptr,
                          nnz * sizeof(th_int_type), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(sorted_col_ptr, col_indices_ptr,
                          nnz * sizeof(th_int_type), cudaMemcpyDeviceToDevice));

    THRUST_CHECK(thrust::sort_by_key(thrust::device,       //
                                     sorted_row_ptr,       // key begin
                                     sorted_row_ptr + nnz, // key end
                                     sorted_col_ptr));

    /////////////////////////////////////////////////////////////////////////
    // Create vals
    /////////////////////////////////////////////////////////////////////////
    th_int_type *unique_row_ptr =
        (th_int_type *)c10::cuda::CUDACachingAllocator::raw_alloc(
            nnz * sizeof(th_int_type));
    scalar_t *reduced_val_ptr =
        (scalar_t *)c10::cuda::CUDACachingAllocator::raw_alloc(
            nnz * sizeof(scalar_t));
    torch::Tensor ones = at::ones({nnz}, mat2.options());
    int num_unique_keys;
    try {
      // reduce by key
      auto end = thrust::reduce_by_key(
          thrust::device,                                // policy
          sorted_row_ptr,                                // key begin
          sorted_row_ptr + nnz,                          // key end
          reinterpret_cast<scalar_t *>(ones.data_ptr()), // value begin
          unique_row_ptr,                                // key out begin
          reduced_val_ptr                                // value out begin
      );
      num_unique_keys = end.first - unique_row_ptr;
      LOG_DEBUG("Num unique keys:", num_unique_keys);
    }
    THRUST_CATCH;

    // Copy the results to the correct output
    inverse_val<th_int_type, scalar_t>
        <<<GET_BLOCKS(nnz, BLOCK_SIZE), BLOCK_SIZE>>>(
            nnz, reinterpret_cast<scalar_t *>(sorted_val.data_ptr()),
            sorted_row_ptr, reduced_val_ptr);

    c10::cuda::CUDACachingAllocator::raw_delete((void *)unique_row_ptr);
    c10::cuda::CUDACachingAllocator::raw_delete((void *)reduced_val_ptr);
    /////////////////////////////////////////////////////////////////////////

    size_t workspace_buffer_size = 0;
    void *workspace_buffer = nullptr;

    cusparseSpMatDescr_t sparse_descr;
    CUSPARSE_CHECK(cusparseCreateCoo(
        &sparse_descr,     //
        dim_i, dim_j, nnz, //
        reinterpret_cast<void *>(sorted_row_ptr),
        reinterpret_cast<void *>(sorted_col_ptr),
        reinterpret_cast<void *>(sorted_val.data_ptr()), //
        std::is_same<th_int_type, int32_t>::value ? CUSPARSE_INDEX_32I
                                                  : CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, cuda_data_type));

    cusparseDnMatDescr_t dense_descr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_descr,                       //
                                       dim_k, dim_j, dim_k,                //
                                       reinterpret_cast<void *>(mat2_ptr), //
                                       cuda_data_type, CUSPARSE_ORDER_COL));

    cusparseDnMatDescr_t result_descr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&result_descr,                        //
                                       dim_i, dim_k, dim_i,                  //
                                       reinterpret_cast<void *>(result_ptr), //
                                       cuda_data_type, CUSPARSE_ORDER_COL));

    size_t required_workspace_buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE, (void *)&alpha_val, sparse_descr,
        dense_descr, (void *)&beta_val, result_descr, cuda_data_type, mm_alg,
        &required_workspace_buffer_size));
    LOG_DEBUG("Buffer size:", required_workspace_buffer_size);

    if (required_workspace_buffer_size > workspace_buffer_size) {
      if (workspace_buffer != nullptr) {
        cudaFree(workspace_buffer);
      }
      workspace_buffer_size = required_workspace_buffer_size;
      LOG_DEBUG("cudaMallocManaged");
      cudaMallocManaged(&workspace_buffer, workspace_buffer_size);
    }
    LOG_DEBUG("SPMM");
    CUSPARSE_CHECK(cusparseSpMM(cusparse_handle,                  //
                                CUSPARSE_OPERATION_NON_TRANSPOSE, //
                                CUSPARSE_OPERATION_TRANSPOSE,     //
                                (void *)&alpha_val,               //
                                sparse_descr, dense_descr,        //
                                (void *)&beta_val, result_descr,  //
                                cuda_data_type, mm_alg, workspace_buffer));
    CUSPARSE_CHECK(cusparseDestroySpMat(sparse_descr));
    CUSPARSE_CHECK(cusparseDestroyDnMat(dense_descr));
    CUSPARSE_CHECK(cusparseDestroyDnMat(result_descr));

    if (workspace_buffer != nullptr) {
      cudaFree(workspace_buffer);
    }
  });

  // Need to transpose the result matrices since cusparse stores
  // them in column-major order in memory
  result.transpose_(0, 1);

  CUDA_CHECK(cudaDeviceSynchronize());

  return {result, sorted_row_col, sorted_val};
}

template std::vector<torch::Tensor> // output, sorted rows_cols, sorted vals.
coo_spmm_average<int32_t>(torch::Tensor const &rows, torch::Tensor const &cols,
                          int64_t const dim_i, int64_t const dim_j,
                          torch::Tensor const &mat2,
                          int64_t const spmm_algorithm_id);

// template torch::Tensor
// coo_spmm<int64_t>(torch::Tensor const &rows, torch::Tensor const &cols,
//                   torch::Tensor const &vals, int64_t const dim_i,
//                   int64_t const dim_j, torch::Tensor const &mat2,
//                   int64_t spmm_algorithm_id);

} // namespace minkowski
