#include <TH/TH.h>
#include <THC/THC.h>

#include <assert.h>
#include <stdbool.h>

#include "ffi/sparse_functions.h"

#define CUDA_CHECK                                                             \
  error = cudaGetLastError();                                                  \
  if (error != cudaSuccess) {                                                  \
    printf("CUDA error: %s\n", cudaGetErrorString(error));                     \
    return -1;                                                                 \
  }

#define INIT_D_DIM_ARRY(TH_ARR, P_ARR)                                         \
  if (THIntTensor_size(TH_ARR, 0) != D) {                                      \
    printf(                                                                    \
        "DSCE ERROR: arrary must be a D-dim vector. size: %ld, expects %ld\n", \
        THIntTensor_size(TH_ARR, 0), D);                                       \
    return -1;                                                                 \
  }                                                                            \
  int *P_ARR = THIntTensor_data(TH_ARR);

#define INIT_COORDS(TH_COORDS, P_COORDS, NROWS)                                \
  int *P_COORDS = THIntTensor_data(TH_COORDS);                                 \
  int NROWS = THIntTensor_size(TH_COORDS, 0);                                  \
  if (THIntTensor_size(TH_COORDS, 1) - 1 != D) {                               \
    printf("DSCE ERROR: coords must be a set of (D + 1)-dim vector\n");        \
    return -1;                                                                 \
  }

#define INIT_OUT_COORDS(SUCCESS, P_PIXEL_DIST, P_STRIDE, IS_TRANSPOSE)         \
  SUCCESS =                                                                    \
      _initialize_out_coords(P_PIXEL_DIST, P_STRIDE, IS_TRANSPOSE, D, m);      \
  if (SUCCESS < 0) {                                                           \
    printf("DSCE ERROR: Failed to initialize output coordinates\n");           \
    return SUCCESS;                                                            \
  }

#define INIT_GLOBAL_COORDS(SUCCESS, P_PIXEL_DIST, BATCH_SIZE)                  \
  SUCCESS = _initialize_origin_coords(P_PIXEL_DIST, BATCH_SIZE, D, m);         \
  if (SUCCESS < 0) {                                                           \
    printf("DSCE ERROR: Failed to initialize output origin coordinates\n");    \
    return SUCCESS;                                                            \
  }

#define GET_OUT_NUM_COORDS(SUCCESS, P_PIXEL_DIST, P_STRIDE, IS_TRANSPOSE,      \
                           NROWS)                                              \
  int *p_out_pixel_dist =                                                      \
      compute_out_pixel_dist(P_PIXEL_DIST, P_STRIDE, D, IS_TRANSPOSE);         \
  SUCCESS = _get_num_coords(p_out_pixel_dist, &NROWS, D, m);                   \
  if (SUCCESS < 0) {                                                           \
    printf("DSCE ERROR: Failed to get output coordinates\n");                  \
    return SUCCESS;                                                            \
  }

#define GET_GLOBAL_OUT_NUM_COORDS(SUCCESS, NROWS)                              \
  int *p_out_pixel_dist = malloc(sizeof(int) * D);                             \
  memset(p_out_pixel_dist, 0, sizeof(int) * D);                                \
  SUCCESS = _get_num_coords(p_out_pixel_dist, &NROWS, D, m);                   \
  if (SUCCESS < 0) {                                                           \
    printf("DSCE ERROR: Failed to get output coordinates\n");                  \
    return SUCCESS;                                                            \
  }

extern THCState *state;

int *compute_out_pixel_dist(int *p_pixel_dist, int *p_stride, long D,
                            bool is_transpose) {
  int *p_out_pixel_dist = malloc(D * sizeof(int));
  int i;
  for (i = 0; i < D; i++) {
    if (is_transpose) {
      p_out_pixel_dist[i] = p_pixel_dist[i] / p_stride[i];
    } else {
      p_out_pixel_dist[i] = p_pixel_dist[i] * p_stride[i];
    }
  }
  return p_out_pixel_dist;
}

long read_ffi_ptr(void **ptr) { return (long)(ptr[0]); }

void write_ffi_ptr(long p, void **ptr) { ptr[0] = (void *)p; }

long initialize_coords(THIntTensor *th_coords, THIntTensor *th_pixel_dist,
                       long D, void **m) {
  INIT_COORDS(th_coords, p_coords, nrows)
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)

  return _initialize_coords(p_coords, nrows, p_pixel_dist, D, m);
}

long initialize_coords_with_duplicates(THIntTensor *th_coords,
                                       THIntTensor *th_pixel_dist, long D,
                                       void **m) {
  INIT_COORDS(th_coords, p_coords, nrows)
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)

  return _initialize_coords_with_duplicates(p_coords, nrows, p_pixel_dist, D,
                                            m);
}

long get_coords(THIntTensor *th_coords, THIntTensor *th_pixel_dist, long D,
                void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)

  long success;
  int nrows = -1;
  success = _get_num_coords(p_pixel_dist, &nrows, D, m);
  if (success < 0) {
    return success;
  }
  // Initialize torch and pass the pointer to fill out data
  THIntTensor_resize2d(th_coords, nrows, D + 1);
  THIntTensor_zero(th_coords);
  int *p_coords = THIntTensor_data(th_coords);
  success = _get_coords(p_coords, p_pixel_dist, D, m);
  return success;
}

long get_permutation(THIntTensor *th_permutation,
                     THIntTensor *th_pixel_dist_src,
                     THIntTensor *th_pixel_dist_dst, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist_src, p_pixel_dist_src)
  INIT_D_DIM_ARRY(th_pixel_dist_dst, p_pixel_dist_dst)

  long success;
  int nrows = -1;
  success = _get_num_coords(p_pixel_dist_dst, &nrows, D, m);
  if (success < 0) {
    return success;
  }
  // Initialize torch and pass the pointer to fill out data
  THIntTensor_resize1d(th_permutation, nrows);
  THIntTensor_zero(th_permutation);
  int *p_permutation = THIntTensor_data(th_permutation);
  success =
      _get_permutation(p_permutation, p_pixel_dist_src, p_pixel_dist_dst, D, m);
  return success;
}

long get_index_map(THIntTensor *th_coords, THIntTensor *th_index_map,
                   THIntTensor *th_pixel_dist, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)

  long success;
  int index_map_nrows = -1;
  success = _get_num_coords(p_pixel_dist, &index_map_nrows, D, m);
  if (success < 0) {
    return success;
  }

  // Set index map
  int nrows = THIntTensor_size(th_coords, 0);
  THIntTensor_resize1d(th_index_map, nrows);
  THIntTensor_zero(th_index_map);
  int *p_coords = THIntTensor_data(th_coords);
  int *p_index_map = THIntTensor_data(th_index_map);

  success = _get_index_map(p_coords, nrows, p_index_map, index_map_nrows,
                           p_pixel_dist, D, m);
  return success;
}

long get_nrows(THIntTensor *th_pixel_dist, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  long success;
  int nrows = -1;

  success = _get_num_coords(p_pixel_dist, &nrows, D, m);
  if (success < 0) {
    return success;
  } else {
    return nrows;
  }
}

void clear(long D, void **m) { _clear(D, m); }

long convolution_forward(THFloatTensor *th_in_feat, THFloatTensor *th_out_feat,
                         THFloatTensor *th_kernel, THIntTensor *th_pixel_dist,
                         THIntTensor *th_stride, THIntTensor *th_kernel_size,
                         THIntTensor *th_dilation, long region_type,
                         THIntTensor *th_offset, uint64_t *p_in_coords_key,
                         uint64_t *p_out_coords_key, long D, void **m) {
  // This will not take coords as the first initialization saved it in metadata
  // th_in_feat is 2D nrows, in_channel
  // th_kernel is 3D filter_volume, in_channel, out_channel
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  long success;
  int in_nchannel, _in_nchannel, out_nchannel, out_nrows = -1;
  int n_offset = 0, *p_offset = NULL;
  bool is_transpose = false;

  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  INIT_OUT_COORDS(success, p_pixel_dist, p_stride, is_transpose)

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  _in_nchannel = THFloatTensor_size(th_kernel, 1);
  out_nchannel = THFloatTensor_size(th_kernel, 2);
  in_nchannel = THFloatTensor_size(th_in_feat, 1);

  // Checks
  if (_in_nchannel != in_nchannel) {
    printf("DSCE ERROR: Kernel channel size and input channel size mismatch\n");
    return -1;
  }

  // Initialize output
  THFloatTensor_resize2d(th_out_feat, out_nrows, out_nchannel);
  THFloatTensor_zero(th_out_feat);

  // Pointers
  float *p_in_feat = THFloatTensor_data(th_in_feat);
  float *p_out_feat = THFloatTensor_data(th_out_feat);
  float *p_kernel = THFloatTensor_data(th_kernel);

  // Custom Region Type
  if (region_type == 2) {
    n_offset = THIntTensor_size(th_offset, 0);
    p_offset = THIntTensor_data(th_offset);
    if (THIntTensor_size(th_offset, 1) != D) {
      printf("DSCE ERROR: Offset size does not match.\n");
      return -1;
    }
  }

  // put exposed variable into _conv_foward;
  return _conv_fw(p_in_feat, in_nchannel, p_out_feat, out_nchannel, p_kernel,
                  out_nrows, p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                  region_type, p_offset, n_offset, p_in_coords_key,
                  p_out_coords_key, D, m);
}

long convolution_transpose_forward(
    THFloatTensor *th_in_feat, THFloatTensor *th_out_feat,
    THFloatTensor *th_kernel, THIntTensor *th_pixel_dist,
    THIntTensor *th_stride, THIntTensor *th_kernel_size,
    THIntTensor *th_dilation, long region_type, THIntTensor *th_offset,
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, long D, void **m) {
  // This will not take coords as the first initialization saved it in metadata
  // th_in_feat is 2D nrows, in_channel
  // th_kernel is 3D filter_volume, in_channel, out_channel
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  long success;
  int in_nchannel, _in_nchannel, out_nchannel, out_nrows = -1;
  int n_offset = 0, *p_offset = NULL;
  bool is_transpose = true;

  // if (pixel_dist % stride != 0) {
  //   printf("DSCE ERROR: pixel distance not divisible by stride.\n");
  //   return -1;
  // }

  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  INIT_OUT_COORDS(success, p_pixel_dist, p_stride, is_transpose)

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  _in_nchannel = THFloatTensor_size(th_kernel, 1);
  out_nchannel = THFloatTensor_size(th_kernel, 2);
  in_nchannel = THFloatTensor_size(th_in_feat, 1);

  // Checks
  if (_in_nchannel != in_nchannel) {
    printf("DSCE ERROR: Kernel channel size and input channel size mismatch\n");
    return -1;
  }

  // Initialize output
  THFloatTensor_resize2d(th_out_feat, out_nrows, out_nchannel);
  THFloatTensor_zero(th_out_feat);

  // Pointers
  float *p_in_feat = THFloatTensor_data(th_in_feat);
  float *p_out_feat = THFloatTensor_data(th_out_feat);
  float *p_kernel = THFloatTensor_data(th_kernel);

  // Custom Region Type
  if (region_type == 2) {
    n_offset = THIntTensor_size(th_offset, 0);
    p_offset = THIntTensor_data(th_offset);
    if (THIntTensor_size(th_offset, 1) != D) {
      printf("DSCE ERROR: Offset size does not match.\n");
      return -1;
    }
  }

  // put exposed variable into _conv_foward;
  return _conv_tr_fw(p_in_feat, in_nchannel, p_out_feat, out_nchannel, p_kernel,
                     out_nrows, p_pixel_dist, p_stride, p_kernel_size,
                     p_dilation, region_type, p_offset, n_offset,
                     p_in_coords_key, p_out_coords_key, D, m);
}

long convolution_backward(THFloatTensor *th_in_feat,
                          THFloatTensor *th_grad_in_feat,
                          THFloatTensor *th_grad_out_feat,
                          THFloatTensor *th_kernel,
                          THFloatTensor *th_grad_kernel,
                          THIntTensor *th_pixel_dist, THIntTensor *th_stride,
                          THIntTensor *th_kernel_size, THIntTensor *th_dilation,
                          uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                          long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  long success;
  int in_nrows, in_nchannel, _in_nchannel, out_nchannel, out_nrows = -1;
  bool is_transpose = false;

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  _in_nchannel = THFloatTensor_size(th_kernel, 1);
  out_nchannel = THFloatTensor_size(th_kernel, 2);
  in_nrows = THFloatTensor_size(th_in_feat, 0);
  in_nchannel = THFloatTensor_size(th_in_feat, 1);

  // Checks
  if (_in_nchannel != in_nchannel) {
    printf("DSCE ERROR: Kernel channel size and input channel size mismatch\n");
    return -1;
  }

  // Initialize output
  THFloatTensor_resize2d(th_grad_in_feat, in_nrows, in_nchannel);
  THFloatTensor_zero(th_grad_in_feat);
  THFloatTensor_resizeAs(th_grad_kernel, th_kernel);
  THFloatTensor_zero(th_grad_kernel);

  // Pointers
  float *p_in_feat = THFloatTensor_data(th_in_feat);
  float *p_grad_in_feat = THFloatTensor_data(th_grad_in_feat);
  float *p_grad_out_feat = THFloatTensor_data(th_grad_out_feat);
  float *p_kernel = THFloatTensor_data(th_kernel);
  float *p_grad_kernel = THFloatTensor_data(th_grad_kernel);

  // put exposed variable into _conv_foward;
  return _conv_bw(p_in_feat, p_grad_in_feat, in_nchannel, p_grad_out_feat,
                  out_nchannel, p_kernel, p_grad_kernel, out_nrows,
                  p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                  p_in_coords_key, p_out_coords_key, D, m);
}

long convolution_transpose_backward(
    THFloatTensor *th_in_feat, THFloatTensor *th_grad_in_feat,
    THFloatTensor *th_grad_out_feat, THFloatTensor *th_kernel,
    THFloatTensor *th_grad_kernel, THIntTensor *th_pixel_dist,
    THIntTensor *th_stride, THIntTensor *th_kernel_size,
    THIntTensor *th_dilation, uint64_t *p_in_coords_key,
    uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  long success;
  int in_nrows, in_nchannel, _in_nchannel, out_nchannel, out_nrows = -1;
  bool is_transpose = true;

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  _in_nchannel = THFloatTensor_size(th_kernel, 1);
  out_nchannel = THFloatTensor_size(th_kernel, 2);
  in_nrows = THFloatTensor_size(th_in_feat, 0);
  in_nchannel = THFloatTensor_size(th_in_feat, 1);

  // Checks
  if (_in_nchannel != in_nchannel) {
    printf("DSCE ERROR: Kernel channel size and input channel size mismatch\n");
    return -1;
  }

  // Initialize output
  THFloatTensor_resize2d(th_grad_in_feat, in_nrows, in_nchannel);
  THFloatTensor_zero(th_grad_in_feat);
  THFloatTensor_resizeAs(th_grad_kernel, th_kernel);
  THFloatTensor_zero(th_grad_kernel);

  // Pointers
  float *p_in_feat = THFloatTensor_data(th_in_feat);
  float *p_grad_in_feat = THFloatTensor_data(th_grad_in_feat);
  float *p_grad_out_feat = THFloatTensor_data(th_grad_out_feat);
  float *p_kernel = THFloatTensor_data(th_kernel);
  float *p_grad_kernel = THFloatTensor_data(th_grad_kernel);

  // put exposed variable into _conv_foward;
  return _conv_tr_bw(p_in_feat, p_grad_in_feat, in_nchannel, p_grad_out_feat,
                     out_nchannel, p_kernel, p_grad_kernel, out_nrows,
                     p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                     p_in_coords_key, p_out_coords_key, D, m);
}

long convolution_forward_gpu(THCudaTensor *th_in_feat,
                             THCudaTensor *th_out_feat, THCudaTensor *th_kernel,
                             THIntTensor *th_pixel_dist, THIntTensor *th_stride,
                             THIntTensor *th_kernel_size,
                             THIntTensor *th_dilation, long region_type,
                             THIntTensor *th_offset, uint64_t *p_in_coords_key,
                             uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  cudaStream_t stream = THCState_getCurrentStream(state);
  long success;
  int in_nchannel, _in_nchannel, out_nchannel, out_nrows = -1;
  int n_offset = 0, *p_offset = NULL;
  bool is_transpose = false;
  cudaError_t error;

  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  INIT_OUT_COORDS(success, p_pixel_dist, p_stride, is_transpose)

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  _in_nchannel = THCudaTensor_size(state, th_kernel, 1);
  out_nchannel = THCudaTensor_size(state, th_kernel, 2);
  in_nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Checks
  if (_in_nchannel != in_nchannel) {
    printf("DSCE ERROR: Kernel channel size and input channel size mismatch\n");
    return -1;
  }

  // Initialize output
  THCudaTensor_resize2d(state, th_out_feat, out_nrows, out_nchannel);
  CUDA_CHECK
  THCudaTensor_zero(state, th_out_feat);

  // Pointers
  float *d_in_feat = THCudaTensor_data(state, th_in_feat);
  float *d_out_feat = THCudaTensor_data(state, th_out_feat);
  float *d_kernel = THCudaTensor_data(state, th_kernel);

  // Custom Region Type
  if (region_type == 2) {
    n_offset = THIntTensor_size(th_offset, 0);
    p_offset = THIntTensor_data(th_offset);
    if (THIntTensor_size(th_offset, 1) != D) {
      printf("DSCE ERROR: Offset size does not match.\n");
      return -1;
    }
  }

  // put exposed variable into _conv_foward;
  return _conv_fw_gpu(
      d_in_feat, in_nchannel, d_out_feat, out_nchannel, d_kernel, out_nrows,
      p_pixel_dist, p_stride, p_kernel_size, p_dilation, region_type, p_offset,
      n_offset, p_in_coords_key, p_out_coords_key, stream, D, m);
}

long convolution_transpose_forward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_out_feat,
    THCudaTensor *th_kernel, THIntTensor *th_pixel_dist, THIntTensor *th_stride,
    THIntTensor *th_kernel_size, THIntTensor *th_dilation, long region_type,
    THIntTensor *th_offset, uint64_t *p_in_coords_key,
    uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  cudaStream_t stream = THCState_getCurrentStream(state);
  long success;
  int in_nchannel, _in_nchannel, out_nchannel, out_nrows = -1;
  int n_offset = 0, *p_offset = NULL;
  cudaError_t error;
  bool is_transpose = true;

  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  INIT_OUT_COORDS(success, p_pixel_dist, p_stride, is_transpose)

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  _in_nchannel = THCudaTensor_size(state, th_kernel, 1);
  out_nchannel = THCudaTensor_size(state, th_kernel, 2);
  in_nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Checks
  if (_in_nchannel != in_nchannel) {
    printf("DSCE ERROR: Kernel channel size and input channel size mismatch\n");
    return -1;
  }

  // Initialize output
  THCudaTensor_resize2d(state, th_out_feat, out_nrows, out_nchannel);
  CUDA_CHECK
  THCudaTensor_zero(state, th_out_feat);

  // Pointers
  float *d_in_feat = THCudaTensor_data(state, th_in_feat);
  float *d_out_feat = THCudaTensor_data(state, th_out_feat);
  float *d_kernel = THCudaTensor_data(state, th_kernel);

  // Custom Region Type
  if (region_type == 2) {
    n_offset = THIntTensor_size(th_offset, 0);
    p_offset = THIntTensor_data(th_offset);
    if (THIntTensor_size(th_offset, 1) != D) {
      printf("DSCE ERROR: Offset size does not match.\n");
      return -1;
    }
  }

  // put exposed variable into _conv_foward;
  return _conv_tr_fw_gpu(
      d_in_feat, in_nchannel, d_out_feat, out_nchannel, d_kernel, out_nrows,
      p_pixel_dist, p_stride, p_kernel_size, p_dilation, region_type, p_offset,
      n_offset, p_in_coords_key, p_out_coords_key, stream, D, m);
}

long convolution_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_grad_out_feat, THCudaTensor *th_kernel,
    THCudaTensor *th_grad_kernel, THIntTensor *th_pixel_dist,
    THIntTensor *th_stride, THIntTensor *th_kernel_size,
    THIntTensor *th_dilation, uint64_t *p_in_coords_key,
    uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)
  cudaStream_t stream = THCState_getCurrentStream(state);
  long success;
  int in_nrows, in_nchannel, _in_nchannel, out_nchannel, out_nrows = -1;
  cudaError_t error;
  bool is_transpose = false;

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  _in_nchannel = THCudaTensor_size(state, th_kernel, 1);
  out_nchannel = THCudaTensor_size(state, th_kernel, 2);
  in_nrows = THCudaTensor_size(state, th_in_feat, 0);
  in_nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Checks
  if (_in_nchannel != in_nchannel) {
    printf("DSCE ERROR: Kernel channel size and input channel size mismatch\n");
    return -1;
  }

  // Initialize output
  THCudaTensor_resize2d(state, th_grad_in_feat, in_nrows, in_nchannel);
  CUDA_CHECK
  THCudaTensor_zero(state, th_grad_in_feat);
  THCudaTensor_resizeAs(state, th_grad_kernel, th_kernel);
  CUDA_CHECK
  THCudaTensor_zero(state, th_grad_kernel);

  // Pointers
  float *d_in_feat = THCudaTensor_data(state, th_in_feat);
  float *d_grad_in_feat = THCudaTensor_data(state, th_grad_in_feat);
  float *d_grad_out_feat = THCudaTensor_data(state, th_grad_out_feat);
  float *d_kernel = THCudaTensor_data(state, th_kernel);
  float *d_grad_kernel = THCudaTensor_data(state, th_grad_kernel);

  // put exposed variable into _conv_foward;
  return _conv_bw_gpu(d_in_feat, d_grad_in_feat, in_nchannel, d_grad_out_feat,
                      out_nchannel, d_kernel, d_grad_kernel, out_nrows,
                      p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                      p_in_coords_key, p_out_coords_key, stream, D, m);
}

long convolution_transpose_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_grad_out_feat, THCudaTensor *th_kernel,
    THCudaTensor *th_grad_kernel, THIntTensor *th_pixel_dist,
    THIntTensor *th_stride, THIntTensor *th_kernel_size,
    THIntTensor *th_dilation, uint64_t *p_in_coords_key,
    uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)
  cudaStream_t stream = THCState_getCurrentStream(state);
  long success;
  int in_nrows, in_nchannel, _in_nchannel, out_nchannel, out_nrows = -1;
  cudaError_t error;
  bool is_transpose = true;

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  _in_nchannel = THCudaTensor_size(state, th_kernel, 1);
  out_nchannel = THCudaTensor_size(state, th_kernel, 2);
  in_nrows = THCudaTensor_size(state, th_in_feat, 0);
  in_nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Checks
  if (_in_nchannel != in_nchannel) {
    printf("DSCE ERROR: Kernel channel size and input channel size mismatch\n");
    return -1;
  }

  // Initialize output
  THCudaTensor_resize2d(state, th_grad_in_feat, in_nrows, in_nchannel);
  CUDA_CHECK
  THCudaTensor_zero(state, th_grad_in_feat);
  THCudaTensor_resizeAs(state, th_grad_kernel, th_kernel);
  CUDA_CHECK
  THCudaTensor_zero(state, th_grad_kernel);

  // Pointers
  float *d_in_feat = THCudaTensor_data(state, th_in_feat);
  float *d_grad_in_feat = THCudaTensor_data(state, th_grad_in_feat);
  float *d_grad_out_feat = THCudaTensor_data(state, th_grad_out_feat);
  float *d_kernel = THCudaTensor_data(state, th_kernel);
  float *d_grad_kernel = THCudaTensor_data(state, th_grad_kernel);

  // put exposed variable into _conv_foward;
  return _conv_tr_bw_gpu(
      d_in_feat, d_grad_in_feat, in_nchannel, d_grad_out_feat, out_nchannel,
      d_kernel, d_grad_kernel, out_nrows, p_pixel_dist, p_stride, p_kernel_size,
      p_dilation, p_in_coords_key, p_out_coords_key, stream, D, m);
}

long max_pooling_forward(THFloatTensor *th_in_feat, THFloatTensor *th_out_feat,
                         THIntTensor *th_mask_index, THIntTensor *th_pixel_dist,
                         THIntTensor *th_stride, THIntTensor *th_kernel_size,
                         THIntTensor *th_dilation, long region_type,
                         THIntTensor *th_offset, uint64_t *p_in_coords_key,
                         uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)
  long success;
  int nchannel, out_nrows = -1;
  int n_offset = 0, *p_offset = NULL;
  bool is_transpose = false;

  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  INIT_OUT_COORDS(success, p_pixel_dist, p_stride, is_transpose)

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  nchannel = THFloatTensor_size(th_in_feat, 1);

  // Initialize output, values will be set within the forward function
  THFloatTensor_resize2d(th_out_feat, out_nrows, nchannel);
  THIntTensor_resize2d(th_mask_index, out_nrows, nchannel);

  // Pointers
  float *p_in_feat = THFloatTensor_data(th_in_feat);
  float *p_out_feat = THFloatTensor_data(th_out_feat);
  int *p_mask_index = THIntTensor_data(th_mask_index);

  // Custom Region Type
  if (region_type == 2) {
    n_offset = THIntTensor_size(th_offset, 0);
    p_offset = THIntTensor_data(th_offset);
    if (THIntTensor_size(th_offset, 1) != D) {
      printf("DSCE ERROR: Offset size does not match.\n");
      return -1;
    }
  }

  // put exposed variable into _conv_foward;
  return _max_pooling_fw(p_in_feat, p_out_feat, p_mask_index, nchannel,
                         out_nrows, p_pixel_dist, p_stride, p_kernel_size,
                         p_dilation, region_type, p_offset, n_offset,
                         p_in_coords_key, p_out_coords_key, D, m);
}

long max_pooling_backward(THFloatTensor *th_in_feat,
                          THFloatTensor *th_grad_in_feat,
                          THFloatTensor *th_grad_out_feat,
                          THIntTensor *th_mask_index,
                          THIntTensor *th_pixel_dist, THIntTensor *th_stride,
                          THIntTensor *th_kernel_size, THIntTensor *th_dilation,
                          uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                          long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)
  long success;
  int in_nrows, nchannel, out_nrows = -1;
  bool is_transpose = false;

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  in_nrows = THFloatTensor_size(th_in_feat, 0);
  nchannel = THFloatTensor_size(th_in_feat, 1);

  // Initialize output
  THFloatTensor_resize2d(th_grad_in_feat, in_nrows, nchannel);
  // THFloatTensor_zero(th_grad_in_feat); set within the function

  // Pointers
  float *p_grad_in_feat = THFloatTensor_data(th_grad_in_feat);
  float *p_grad_out_feat = THFloatTensor_data(th_grad_out_feat);
  int *p_mask_index = THIntTensor_data(th_mask_index);

  // put exposed variable into _conv_foward;
  return _max_pooling_bw(p_grad_in_feat, in_nrows, p_grad_out_feat, out_nrows,
                         p_mask_index, nchannel, p_pixel_dist, p_stride,
                         p_kernel_size, p_dilation, p_in_coords_key,
                         p_out_coords_key, D, m);
}

long max_pooling_forward_gpu(THCudaTensor *th_in_feat,
                             THCudaTensor *th_out_feat,
                             THCudaIntTensor *th_mask_index,
                             THIntTensor *th_pixel_dist, THIntTensor *th_stride,
                             THIntTensor *th_kernel_size,
                             THIntTensor *th_dilation, long region_type,
                             THIntTensor *th_offset, uint64_t *p_in_coords_key,
                             uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  cudaStream_t stream = THCState_getCurrentStream(state);
  long success;
  int nchannel, out_nrows = -1;
  int n_offset = 0, *p_offset = NULL;
  bool is_transpose = false;

  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  INIT_OUT_COORDS(success, p_pixel_dist, p_stride, is_transpose)

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Initialize output, values will be set within the forward function
  THCudaTensor_resize2d(state, th_out_feat, out_nrows, nchannel);
  THCudaIntTensor_resize2d(state, th_mask_index, out_nrows, nchannel);

  // Pointers
  float *d_in_feat = THCudaTensor_data(state, th_in_feat);
  float *d_out_feat = THCudaTensor_data(state, th_out_feat);
  int *d_mask_index = THCudaIntTensor_data(state, th_mask_index);

  // Custom Region Type
  if (region_type == 2) {
    n_offset = THIntTensor_size(th_offset, 0);
    p_offset = THIntTensor_data(th_offset);
    if (THIntTensor_size(th_offset, 1) != D) {
      printf("DSCE ERROR: Offset size does not match.\n");
      return -1;
    }
  }

  // put exposed variable into _conv_foward;
  return _max_pooling_fw_gpu(d_in_feat, d_out_feat, out_nrows, d_mask_index,
                             nchannel, p_pixel_dist, p_stride, p_kernel_size,
                             p_dilation, region_type, p_offset, n_offset,
                             p_in_coords_key, p_out_coords_key, stream, D, m);
}

long max_pooling_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_grad_out_feat, THCudaIntTensor *th_mask_index,
    THIntTensor *th_pixel_dist, THIntTensor *th_stride,
    THIntTensor *th_kernel_size, THIntTensor *th_dilation,
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  cudaStream_t stream = THCState_getCurrentStream(state);
  long success;
  int in_nrows, nchannel, out_nrows = -1;
  bool is_transpose = false;

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  in_nrows = THCudaTensor_size(state, th_in_feat, 0);
  nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Initialize output
  THCudaTensor_resize2d(state, th_grad_in_feat, in_nrows, nchannel);
  // THFloatTensor_zero(th_grad_in_feat); set within the function

  // Pointers
  float *d_grad_in_feat = THCudaTensor_data(state, th_grad_in_feat);
  float *d_grad_out_feat = THCudaTensor_data(state, th_grad_out_feat);
  int *d_mask_index = THCudaIntTensor_data(state, th_mask_index);

  // put exposed variable into _conv_foward;
  return _max_pooling_bw_gpu(d_grad_in_feat, in_nrows, d_grad_out_feat,
                             out_nrows, d_mask_index, nchannel, p_pixel_dist,
                             p_stride, p_kernel_size, p_dilation,
                             p_in_coords_key, p_out_coords_key, stream, D, m);
}

// Nonzero avg
long nonzero_avg_pooling_forward(
    THFloatTensor *th_in_feat, THFloatTensor *th_out_feat,
    THFloatTensor *th_num_nonzero, THIntTensor *th_pixel_dist,
    THIntTensor *th_stride, THIntTensor *th_kernel_size,
    THIntTensor *th_dilation, long region_type, THIntTensor *th_offset,
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)
  long success;
  int nchannel, out_nrows = -1;
  int n_offset = 0, *p_offset = NULL;
  bool is_transpose = false;

  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  INIT_OUT_COORDS(success, p_pixel_dist, p_stride, is_transpose)

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  nchannel = THFloatTensor_size(th_in_feat, 1);

  // Initialize output, values will be set within the forward function
  THFloatTensor_resize2d(th_out_feat, out_nrows, nchannel);
  THFloatTensor_resize1d(th_num_nonzero, out_nrows);

  // Pointers
  float *p_in_feat = THFloatTensor_data(th_in_feat);
  float *p_out_feat = THFloatTensor_data(th_out_feat);
  float *p_num_nonzero = THFloatTensor_data(th_num_nonzero);

  // Custom Region Type
  if (region_type == 2) {
    n_offset = THIntTensor_size(th_offset, 0);
    p_offset = THIntTensor_data(th_offset);
    if (THIntTensor_size(th_offset, 1) != D) {
      printf("DSCE ERROR: Offset size does not match.\n");
      return -1;
    }
  }

  // put exposed variable into _conv_foward;
  return _nonzero_avg_pooling_fw(
      p_in_feat, p_out_feat, p_num_nonzero, nchannel, out_nrows, p_pixel_dist,
      p_stride, p_kernel_size, p_dilation, region_type, p_offset, n_offset,
      p_in_coords_key, p_out_coords_key, D, m);
}

long nonzero_avg_pooling_backward(
    THFloatTensor *th_in_feat, THFloatTensor *th_grad_in_feat,
    THFloatTensor *th_grad_out_feat, THFloatTensor *th_num_nonzero,
    THIntTensor *th_pixel_dist, THIntTensor *th_stride,
    THIntTensor *th_kernel_size, THIntTensor *th_dilation,
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)
  long success;
  int in_nrows, nchannel, out_nrows = -1;
  bool is_transpose = false;

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  in_nrows = THFloatTensor_size(th_in_feat, 0);
  nchannel = THFloatTensor_size(th_in_feat, 1);

  // Initialize output
  THFloatTensor_resize2d(th_grad_in_feat, in_nrows, nchannel);
  // THFloatTensor_zero(th_grad_in_feat); set within the function

  // Pointers
  float *p_grad_in_feat = THFloatTensor_data(th_grad_in_feat);
  float *p_grad_out_feat = THFloatTensor_data(th_grad_out_feat);
  float *p_num_nonzero = THFloatTensor_data(th_num_nonzero);

  // put exposed variable into _conv_foward;
  return _nonzero_avg_pooling_bw(
      p_grad_in_feat, in_nrows, p_grad_out_feat, out_nrows, p_num_nonzero,
      nchannel, p_pixel_dist, p_stride, p_kernel_size, p_dilation,
      p_in_coords_key, p_out_coords_key, D, m);
}

long nonzero_avg_pooling_forward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_out_feat,
    THCudaTensor *th_num_nonzero, THIntTensor *th_pixel_dist,
    THIntTensor *th_stride, THIntTensor *th_kernel_size,
    THIntTensor *th_dilation, long region_type, THIntTensor *th_offset,
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  cudaStream_t stream = THCState_getCurrentStream(state);
  long success;
  int nchannel, in_nrows, out_nrows = -1;
  int n_offset = 0, *p_offset = NULL;
  bool is_transpose = false;

  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  INIT_OUT_COORDS(success, p_pixel_dist, p_stride, is_transpose)

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  in_nrows = THCudaTensor_size(state, th_in_feat, 0);
  nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Initialize output, values will be set within the forward function
  THCudaTensor_resize2d(state, th_out_feat, out_nrows, nchannel);
  THCudaTensor_resize1d(state, th_num_nonzero, out_nrows);

  // Pointers
  float *d_in_feat = THCudaTensor_data(state, th_in_feat);
  float *d_out_feat = THCudaTensor_data(state, th_out_feat);
  float *d_num_nonzero = THCudaTensor_data(state, th_num_nonzero);

  // Custom Region Type
  if (region_type == 2) {
    n_offset = THIntTensor_size(th_offset, 0);
    p_offset = THIntTensor_data(th_offset);
    if (THIntTensor_size(th_offset, 1) != D) {
      printf("DSCE ERROR: Offset size does not match.\n");
      return -1;
    }
  }

  // put exposed variable into _conv_foward;
  return _nonzero_avg_pooling_fw_gpu(
      d_in_feat, in_nrows, d_out_feat, out_nrows, d_num_nonzero, nchannel,
      p_pixel_dist, p_stride, p_kernel_size, p_dilation, region_type, p_offset,
      n_offset, p_in_coords_key, p_out_coords_key, stream, D, m);
}

long nonzero_avg_pooling_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_grad_out_feat, THCudaTensor *th_num_nonzero,
    THIntTensor *th_pixel_dist, THIntTensor *th_stride,
    THIntTensor *th_kernel_size, THIntTensor *th_dilation,
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  cudaStream_t stream = THCState_getCurrentStream(state);
  long success;
  int in_nrows, nchannel, out_nrows = -1;
  bool is_transpose = false;

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  in_nrows = THCudaTensor_size(state, th_in_feat, 0);
  nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Initialize output
  THCudaTensor_resize2d(state, th_grad_in_feat, in_nrows, nchannel);
  // THFloatTensor_zero(th_grad_in_feat); set within the function

  // Pointers
  float *d_grad_in_feat = THCudaTensor_data(state, th_grad_in_feat);
  float *d_grad_out_feat = THCudaTensor_data(state, th_grad_out_feat);
  float *d_num_nonzero = THCudaTensor_data(state, th_num_nonzero);

  // put exposed variable into _conv_foward;
  return _nonzero_avg_pooling_bw_gpu(
      d_grad_in_feat, in_nrows, d_grad_out_feat, out_nrows, d_num_nonzero,
      nchannel, p_pixel_dist, p_stride, p_kernel_size, p_dilation,
      p_in_coords_key, p_out_coords_key, stream, D, m);
}

// Unpooling
long unpooling_forward(THFloatTensor *th_in_feat, THFloatTensor *th_out_feat,
                       THFloatTensor *th_num_nonzero,
                       THIntTensor *th_pixel_dist, THIntTensor *th_stride,
                       THIntTensor *th_kernel_size, THIntTensor *th_dilation,
                       long region_type, THIntTensor *th_offset,
                       uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                       long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  long success;
  int nchannel, out_nrows = -1;
  int n_offset = 0, *p_offset = NULL;
  bool is_transpose = true;

  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  INIT_OUT_COORDS(success, p_pixel_dist, p_stride, is_transpose)

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  nchannel = THFloatTensor_size(th_in_feat, 1);

  // Initialize output, values will be set within the forward function
  THFloatTensor_resize2d(th_out_feat, out_nrows, nchannel);
  THFloatTensor_resize1d(th_num_nonzero, out_nrows);

  // Pointers
  float *p_in_feat = THFloatTensor_data(th_in_feat);
  float *p_out_feat = THFloatTensor_data(th_out_feat);
  float *p_num_nonzero = THFloatTensor_data(th_num_nonzero);

  // Custom Region Type
  if (region_type == 2) {
    n_offset = THIntTensor_size(th_offset, 0);
    p_offset = THIntTensor_data(th_offset);
    if (THIntTensor_size(th_offset, 1) != D) {
      printf("DSCE ERROR: Offset size does not match.\n");
      return -1;
    }
  }

  // put exposed variable into _conv_foward;
  return _unpooling_fw(p_in_feat, p_out_feat, p_num_nonzero, nchannel,
                       out_nrows, p_pixel_dist, p_stride, p_kernel_size,
                       p_dilation, region_type, p_offset, n_offset,
                       p_in_coords_key, p_out_coords_key, D, m);
}

long unpooling_backward(THFloatTensor *th_in_feat,
                        THFloatTensor *th_grad_in_feat,
                        THFloatTensor *th_grad_out_feat,
                        THFloatTensor *th_num_nonzero,
                        THIntTensor *th_pixel_dist, THIntTensor *th_stride,
                        THIntTensor *th_kernel_size, THIntTensor *th_dilation,
                        uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                        long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)
  long success;
  int in_nrows, nchannel, out_nrows = -1;
  bool is_transpose = true;

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  in_nrows = THFloatTensor_size(th_in_feat, 0);
  nchannel = THFloatTensor_size(th_in_feat, 1);

  // Initialize output
  THFloatTensor_resize2d(th_grad_in_feat, in_nrows, nchannel);
  // THFloatTensor_zero(th_grad_in_feat); set within the function

  // Pointers
  float *p_grad_in_feat = THFloatTensor_data(th_grad_in_feat);
  float *p_grad_out_feat = THFloatTensor_data(th_grad_out_feat);
  float *p_num_nonzero = THFloatTensor_data(th_num_nonzero);

  // put exposed variable into _conv_foward;
  return _unpooling_bw(p_grad_in_feat, in_nrows, p_grad_out_feat, out_nrows,
                       p_num_nonzero, nchannel, p_pixel_dist, p_stride,
                       p_kernel_size, p_dilation, p_in_coords_key,
                       p_out_coords_key, D, m);
}

long unpooling_forward_gpu(THCudaTensor *th_in_feat, THCudaTensor *th_out_feat,
                           THCudaTensor *th_num_nonzero,
                           THIntTensor *th_pixel_dist, THIntTensor *th_stride,
                           THIntTensor *th_kernel_size,
                           THIntTensor *th_dilation, long region_type,
                           THIntTensor *th_offset, uint64_t *p_in_coords_key,
                           uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  cudaStream_t stream = THCState_getCurrentStream(state);
  long success;
  int nchannel, in_nrows, out_nrows = -1;
  int n_offset = 0, *p_offset = NULL;
  bool is_transpose = true;

  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  INIT_OUT_COORDS(success, p_pixel_dist, p_stride, is_transpose)

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  in_nrows = THCudaTensor_size(state, th_in_feat, 0);
  nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Initialize output, values will be set within the forward function
  THCudaTensor_resize2d(state, th_out_feat, out_nrows, nchannel);
  THCudaTensor_resize1d(state, th_num_nonzero, out_nrows);

  // Pointers
  float *d_in_feat = THCudaTensor_data(state, th_in_feat);
  float *d_out_feat = THCudaTensor_data(state, th_out_feat);
  float *d_num_nonzero = THCudaTensor_data(state, th_num_nonzero);

  // Custom Region Type
  if (region_type == 2) {
    n_offset = THIntTensor_size(th_offset, 0);
    p_offset = THIntTensor_data(th_offset);
    if (THIntTensor_size(th_offset, 1) != D) {
      printf("DSCE ERROR: Offset size does not match.\n");
      return -1;
    }
  }

  // put exposed variable into _conv_foward;
  return _unpooling_fw_gpu(
      d_in_feat, in_nrows, d_out_feat, out_nrows, d_num_nonzero, nchannel,
      p_pixel_dist, p_stride, p_kernel_size, p_dilation, region_type, p_offset,
      n_offset, p_in_coords_key, p_out_coords_key, stream, D, m);
}

long unpooling_backward_gpu(THCudaTensor *th_in_feat,
                            THCudaTensor *th_grad_in_feat,
                            THCudaTensor *th_grad_out_feat,
                            THCudaTensor *th_num_nonzero,
                            THIntTensor *th_pixel_dist, THIntTensor *th_stride,
                            THIntTensor *th_kernel_size,
                            THIntTensor *th_dilation, uint64_t *p_in_coords_key,
                            uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  cudaStream_t stream = THCState_getCurrentStream(state);
  long success;
  int in_nrows, nchannel, out_nrows = -1;
  bool is_transpose = true;

  // Get the number of rows required to initialize the th_out_tensor
  GET_OUT_NUM_COORDS(success, p_pixel_dist, p_stride, is_transpose, out_nrows)

  // expose all variables and resize out tensor
  in_nrows = THCudaTensor_size(state, th_in_feat, 0);
  nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Initialize output
  THCudaTensor_resize2d(state, th_grad_in_feat, in_nrows, nchannel);
  // THFloatTensor_zero(th_grad_in_feat); set within the function

  // Pointers
  float *d_grad_in_feat = THCudaTensor_data(state, th_grad_in_feat);
  float *d_grad_out_feat = THCudaTensor_data(state, th_grad_out_feat);
  float *d_num_nonzero = THCudaTensor_data(state, th_num_nonzero);

  // put exposed variable into _conv_foward;
  return _unpooling_bw_gpu(d_grad_in_feat, in_nrows, d_grad_out_feat, out_nrows,
                           d_num_nonzero, nchannel, p_pixel_dist, p_stride,
                           p_kernel_size, p_dilation, p_in_coords_key,
                           p_out_coords_key, stream, D, m);
}

long global_avg_pooling_forward(THFloatTensor *th_in_feat,
                                THFloatTensor *th_out_feat,
                                THFloatTensor *th_num_nonzero,
                                THIntTensor *th_pixel_dist, long batch_size,
                                uint64_t *p_in_coords_key,
                                uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  long success;
  int nchannel, out_nrows = -1;

  INIT_GLOBAL_COORDS(success, p_pixel_dist, batch_size)
  GET_GLOBAL_OUT_NUM_COORDS(success, out_nrows)

  // expose all variables and resize out tensor
  nchannel = THFloatTensor_size(th_in_feat, 1);

  // Initialize output, values will be set within the forward function
  THFloatTensor_resize2d(th_out_feat, out_nrows, nchannel);
  THFloatTensor_resize1d(th_num_nonzero, out_nrows);

  // Pointers
  float *p_in_feat = THFloatTensor_data(th_in_feat);
  float *p_out_feat = THFloatTensor_data(th_out_feat);
  float *p_num_nonzero = THFloatTensor_data(th_num_nonzero);

  // put exposed variable into _conv_foward;
  return _global_avg_pooling_fw(p_in_feat, p_out_feat, out_nrows, nchannel,
                                p_num_nonzero, p_pixel_dist, p_in_coords_key,
                                p_out_coords_key, D, m);
}

long global_avg_pooling_backward(THFloatTensor *th_in_feat,
                                 THFloatTensor *th_grad_in_feat,
                                 THFloatTensor *th_grad_out_feat,
                                 THFloatTensor *th_num_nonzero,
                                 THIntTensor *th_pixel_dist,
                                 uint64_t *p_in_coords_key,
                                 uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  long success;
  int in_nrows, nchannel, out_nrows = -1;

  // Get the number of rows required to initialize the th_out_tensor
  GET_GLOBAL_OUT_NUM_COORDS(success, out_nrows)

  // expose all variables and resize out tensor
  in_nrows = THFloatTensor_size(th_in_feat, 0);
  nchannel = THFloatTensor_size(th_in_feat, 1);

  // Initialize output
  THFloatTensor_resize2d(th_grad_in_feat, in_nrows, nchannel);

  // Pointers
  float *p_grad_in_feat = THFloatTensor_data(th_grad_in_feat);
  float *p_grad_out_feat = THFloatTensor_data(th_grad_out_feat);
  float *p_num_nonzero = THFloatTensor_data(th_num_nonzero);

  // put exposed variable into _conv_foward;
  return _global_avg_pooling_bw(
      p_grad_in_feat, in_nrows, p_grad_out_feat, out_nrows, nchannel,
      p_num_nonzero, p_pixel_dist, p_in_coords_key, p_out_coords_key, D, m);
}

long global_avg_pooling_forward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_out_feat,
    THCudaTensor *th_num_nonzero, THIntTensor *th_pixel_dist, long batch_size,
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)

  cudaStream_t stream = THCState_getCurrentStream(state);
  long success;
  int nchannel, in_nrows, out_nrows = -1;

  INIT_GLOBAL_COORDS(success, p_pixel_dist, batch_size)
  GET_GLOBAL_OUT_NUM_COORDS(success, out_nrows)

  // expose all variables and resize out tensor
  in_nrows = THCudaTensor_size(state, th_in_feat, 0);
  nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Initialize output, values will be set within the forward function
  THCudaTensor_resize2d(state, th_out_feat, out_nrows, nchannel);
  THCudaTensor_resize1d(state, th_num_nonzero, out_nrows);

  // Pointers
  float *d_in_feat = THCudaTensor_data(state, th_in_feat);
  float *d_out_feat = THCudaTensor_data(state, th_out_feat);
  float *d_num_nonzero = THCudaTensor_data(state, th_num_nonzero);

  // put exposed variable into _conv_foward;
  return _global_avg_pooling_fw_gpu(
      d_in_feat, in_nrows, d_out_feat, out_nrows, nchannel, d_num_nonzero,
      p_pixel_dist, p_in_coords_key, p_out_coords_key, stream, D, m);
}

long global_avg_pooling_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_grad_out_feat, THCudaTensor *th_num_nonzero,
    THIntTensor *th_pixel_dist, uint64_t *p_in_coords_key,
    uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  cudaStream_t stream = THCState_getCurrentStream(state);
  long success;
  int in_nrows, nchannel, out_nrows = -1;

  // Get the number of rows required to initialize the th_out_tensor
  GET_GLOBAL_OUT_NUM_COORDS(success, out_nrows)

  // expose all variables and resize out tensor
  in_nrows = THCudaTensor_size(state, th_in_feat, 0);
  nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Initialize output
  THCudaTensor_resize2d(state, th_grad_in_feat, in_nrows, nchannel);

  // Pointers
  float *d_grad_in_feat = THCudaTensor_data(state, th_grad_in_feat);
  float *d_grad_out_feat = THCudaTensor_data(state, th_grad_out_feat);
  float *d_num_nonzero = THCudaTensor_data(state, th_num_nonzero);

  // put exposed variable into _conv_foward;
  return _global_avg_pooling_bw_gpu(d_grad_in_feat, in_nrows, d_grad_out_feat,
                                    out_nrows, nchannel, d_num_nonzero,
                                    p_pixel_dist, p_in_coords_key,
                                    p_out_coords_key, stream, D, m);
}

long global_broadcast_forward(THFloatTensor *th_in_feat,
                              THFloatTensor *th_in_feat_global,
                              THFloatTensor *th_out_feat,
                              THIntTensor *th_pixel_dist, long op,
                              uint64_t *p_in_coords_key,
                              uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  long success;
  int nchannel, in_nrows, in_nrows_global = -1;

  GET_GLOBAL_OUT_NUM_COORDS(success, in_nrows_global)

  // expose all variables and resize out tensor
  in_nrows = THFloatTensor_size(th_in_feat, 0);
  nchannel = THFloatTensor_size(th_in_feat, 1);

  // Initialize output, values will be set within the forward function
  THFloatTensor_resize2d(th_out_feat, in_nrows, nchannel);

  // Pointers
  float *p_in_feat = THFloatTensor_data(th_in_feat);
  float *p_in_feat_global = THFloatTensor_data(th_in_feat_global);
  float *p_out_feat = THFloatTensor_data(th_out_feat);

  // put exposed variable into _conv_foward;
  return _global_broadcast_fw(
      p_in_feat, in_nrows, p_in_feat_global, in_nrows_global, p_out_feat,
      nchannel, p_pixel_dist, op, p_in_coords_key, p_out_coords_key, D, m);
}

long global_broadcast_backward(
    THFloatTensor *th_in_feat, THFloatTensor *th_grad_in_feat,
    THFloatTensor *th_in_feat_global, THFloatTensor *th_grad_in_feat_global,
    THFloatTensor *th_grad_out_feat, THIntTensor *th_pixel_dist, long op,
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  long success;
  int nchannel, in_nrows, in_nrows_global = -1;

  GET_GLOBAL_OUT_NUM_COORDS(success, in_nrows_global)

  // expose all variables and resize out tensor
  in_nrows = THFloatTensor_size(th_in_feat, 0);
  nchannel = THFloatTensor_size(th_in_feat, 1);

  // Initialize output, values will be set within the forward function
  THFloatTensor_resize2d(th_grad_in_feat, in_nrows, nchannel);
  THFloatTensor_resize2d(th_grad_in_feat_global, in_nrows_global, nchannel);

  // Pointers
  float *p_in_feat = THFloatTensor_data(th_in_feat);
  float *p_grad_in_feat = THFloatTensor_data(th_grad_in_feat);
  float *p_in_feat_global = THFloatTensor_data(th_in_feat_global);
  float *p_grad_in_feat_global = THFloatTensor_data(th_grad_in_feat_global);
  float *p_grad_out_feat = THFloatTensor_data(th_grad_out_feat);

  // put exposed variable into _conv_foward;
  return _global_broadcast_bw(
      p_in_feat, p_grad_in_feat, in_nrows, p_in_feat_global,
      p_grad_in_feat_global, in_nrows_global, p_grad_out_feat, nchannel,
      p_pixel_dist, op, p_in_coords_key, p_out_coords_key, D, m);
}

long global_broadcast_forward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_in_feat_global,
    THCudaTensor *th_out_feat, THIntTensor *th_pixel_dist, long op,
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  cudaStream_t stream = THCState_getCurrentStream(state);
  long success;
  int nchannel, in_nrows, in_nrows_global = -1;

  GET_GLOBAL_OUT_NUM_COORDS(success, in_nrows_global)

  // expose all variables and resize out tensor
  in_nrows = THCudaTensor_size(state, th_in_feat, 0);
  nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Initialize output, values will be set within the forward function
  THCudaTensor_resize2d(state, th_out_feat, in_nrows, nchannel);

  // Pointers
  float *d_in_feat = THCudaTensor_data(state, th_in_feat);
  float *d_in_feat_global = THCudaTensor_data(state, th_in_feat_global);
  float *d_out_feat = THCudaTensor_data(state, th_out_feat);

  // put exposed variable into _conv_foward;
  return _global_broadcast_fw_gpu(d_in_feat, in_nrows, d_in_feat_global,
                                  in_nrows_global, d_out_feat, nchannel,
                                  p_pixel_dist, op, p_in_coords_key,
                                  p_out_coords_key, stream, D, m);
}

long global_broadcast_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_in_feat_global, THCudaTensor *th_grad_in_feat_global,
    THCudaTensor *th_grad_out_feat, THIntTensor *th_pixel_dist, long op,
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  cudaStream_t stream = THCState_getCurrentStream(state);
  long success;
  int nchannel, in_nrows, in_nrows_global = -1;

  GET_GLOBAL_OUT_NUM_COORDS(success, in_nrows_global)

  // expose all variables and resize out tensor
  in_nrows = THCudaTensor_size(state, th_in_feat, 0);
  nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Initialize output, values will be set within the forward function
  THCudaTensor_resize2d(state, th_grad_in_feat, in_nrows, nchannel);
  THCudaTensor_resize2d(state, th_grad_in_feat_global, in_nrows_global,
                        nchannel);

  // Pointers
  float *d_in_feat = THCudaTensor_data(state, th_in_feat);
  float *d_grad_in_feat = THCudaTensor_data(state, th_grad_in_feat);
  float *d_in_feat_global = THCudaTensor_data(state, th_in_feat_global);
  float *d_grad_in_feat_global =
      THCudaTensor_data(state, th_grad_in_feat_global);
  float *d_grad_out_feat = THCudaTensor_data(state, th_grad_out_feat);

  // put exposed variable into _conv_foward;
  return _global_broadcast_bw_gpu(
      d_in_feat, d_grad_in_feat, in_nrows, d_in_feat_global,
      d_grad_in_feat_global, in_nrows_global, d_grad_out_feat, nchannel,
      p_pixel_dist, op, p_in_coords_key, p_out_coords_key, stream, D, m);
}
