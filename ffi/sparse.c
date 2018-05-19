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
  if (THLongTensor_size(TH_ARR, 0) != D) {                                     \
    printf("DSCE ERROR: arrary must be a D-dim vector");                       \
    return -1;                                                                 \
  }                                                                            \
  long *P_ARR = THLongTensor_data(TH_ARR);

#define INIT_COORDS(TH_COORDS, P_COORDS, NROWS)                                \
  long *P_COORDS = THLongTensor_data(TH_COORDS);                               \
  int NROWS = THLongTensor_size(TH_COORDS, 0);                                 \
  if (THLongTensor_size(TH_COORDS, 1) - 1 != D) {                              \
    printf("DSCE ERROR: coords must be a set of (D + 1)-dim vector");          \
    return -1;                                                                 \
  }

#define INIT_OUT_COORDS(SUCCESS, P_PIXEL_DIST, P_STRIDE, IS_TRANSPOSE)         \
  SUCCESS =                                                                    \
      _initialize_out_coords(P_PIXEL_DIST, P_STRIDE, IS_TRANSPOSE, D, m);      \
  if (SUCCESS < 0) {                                                           \
    printf("DSCE ERROR: Failed to initialize output coordinates");             \
    return SUCCESS;                                                            \
  }

#define GET_OUT_NUM_COORDS(SUCCESS, P_PIXEL_DIST, P_STRIDE, IS_TRANSPOSE,      \
                           NROWS)                                              \
  long *p_out_pixel_dist =                                                     \
      compute_out_pixel_dist(P_PIXEL_DIST, P_STRIDE, D, IS_TRANSPOSE);         \
  SUCCESS = _get_num_coords(p_out_pixel_dist, &NROWS, D, m);                   \
  if (SUCCESS < 0) {                                                           \
    printf("DSCE ERROR: Failed to get output coordinates");                    \
    return SUCCESS;                                                            \
  }

extern THCState *state;

long *compute_out_pixel_dist(long *p_pixel_dist, long *p_stride, long D,
                             bool is_transpose) {
  long *p_out_pixel_dist = malloc(D * sizeof(long));
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

long initialize_coords(THLongTensor *th_coords, THLongTensor *th_pixel_dist,
                       long D, void **m) {
  INIT_COORDS(th_coords, p_coords, nrows)
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)

  return _initialize_coords(p_coords, nrows, p_pixel_dist, D, m);
}

long initialize_coords_with_duplicates(THLongTensor *th_coords,
                                       THLongTensor *th_pixel_dist, long D,
                                       void **m) {
  INIT_COORDS(th_coords, p_coords, nrows)
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)

  return _initialize_coords_with_duplicates(p_coords, nrows, p_pixel_dist, D,
                                            m);
}

long get_coords(THLongTensor *th_coords, THLongTensor *th_pixel_dist, long D,
                void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)

  long success, nrows = -1;
  success = _get_num_coords(p_pixel_dist, &nrows, D, m);
  if (success < 0) {
    return success;
  }
  // Initialize torch and pass the pointer to fill out data
  THLongTensor_resize2d(th_coords, nrows, D + 1);
  THLongTensor_zero(th_coords);
  long *p_coords = THLongTensor_data(th_coords);
  success = _get_coords(p_coords, p_pixel_dist, D, m);
  return success;
}

long get_permutation(THLongTensor *th_permutation,
                     THLongTensor *th_pixel_dist_src,
                     THLongTensor *th_pixel_dist_dst, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist_src, p_pixel_dist_src)
  INIT_D_DIM_ARRY(th_pixel_dist_dst, p_pixel_dist_dst)

  long success, nrows = -1;
  success = _get_num_coords(p_pixel_dist_dst, &nrows, D, m);
  if (success < 0) {
    return success;
  }
  // Initialize torch and pass the pointer to fill out data
  THLongTensor_resize1d(th_permutation, nrows);
  THLongTensor_zero(th_permutation);
  long *p_permutation = THLongTensor_data(th_permutation);
  success =
      _get_permutation(p_permutation, p_pixel_dist_src, p_pixel_dist_dst, D, m);
  return success;
}

long get_index_map(THLongTensor *th_coords, THLongTensor *th_index_map,
                   THLongTensor *th_pixel_dist, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)

  long success, index_map_nrows = -1;
  success = _get_num_coords(p_pixel_dist, &index_map_nrows, D, m);
  if (success < 0) {
    return success;
  }

  // Set index map
  long nrows = THLongTensor_size(th_coords, 0);
  THLongTensor_resize1d(th_index_map, nrows);
  THLongTensor_zero(th_index_map);
  long *p_coords = THLongTensor_data(th_coords);
  long *p_index_map = THLongTensor_data(th_index_map);

  success = _get_index_map(p_coords, nrows, p_index_map, index_map_nrows,
                           p_pixel_dist, D, m);
  return success;
}

long get_nrows(THLongTensor *th_pixel_dist, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  long success, nrows = -1;

  success = _get_num_coords(p_pixel_dist, &nrows, D, m);
  if (success < 0) {
    return success;
  } else {
    return nrows;
  }
}

void clear(long D, void **m) { _clear(D, m); }

long convolution_forward(THFloatTensor *th_in_feat, THFloatTensor *th_out_feat,
                         THFloatTensor *th_kernel, THLongTensor *th_pixel_dist,
                         THLongTensor *th_stride, THLongTensor *th_kernel_size,
                         THLongTensor *th_dilation, long region_type,
                         THLongTensor *th_offset, long D, void **m) {
  // This will not take coords as the first initialization saved it in metadata
  // th_in_feat is 2D nrows, in_channel
  // th_kernel is 3D filter_volume, in_channel, out_channel
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  long in_nchannel, _in_nchannel, out_nchannel, success, out_nrows = -1;
  long n_offset = 0, *p_offset = NULL;
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
    printf("DSCE ERROR: Kernel channel size and input channel size mismatch");
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
    n_offset = THLongTensor_size(th_offset, 0);
    p_offset = THLongTensor_data(th_offset);
    if (THLongTensor_size(th_offset, 1) != D) {
      printf("DSCE ERROR: Offset size does not match.");
      return -1;
    }
  }

  // put exposed variable into _conv_foward;
  return _conv_fw(p_in_feat, in_nchannel, p_out_feat, out_nchannel, p_kernel,
                  out_nrows, p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                  region_type, p_offset, n_offset, D, m);
}

long convolution_transpose_forward(THFloatTensor *th_in_feat,
                                   THFloatTensor *th_out_feat,
                                   THFloatTensor *th_kernel,
                                   THLongTensor *th_pixel_dist,
                                   THLongTensor *th_stride,
                                   THLongTensor *th_kernel_size,
                                   THLongTensor *th_dilation, long region_type,
                                   THLongTensor *th_offset, long D, void **m) {
  // This will not take coords as the first initialization saved it in metadata
  // th_in_feat is 2D nrows, in_channel
  // th_kernel is 3D filter_volume, in_channel, out_channel
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  long in_nchannel, _in_nchannel, out_nchannel, success, out_nrows = -1;
  long n_offset = 0, *p_offset = NULL;
  bool is_transpose = true;

  // if (pixel_dist % stride != 0) {
  //   printf("DSCE ERROR: pixel distance not divisible by stride.");
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
    printf("DSCE ERROR: Kernel channel size and input channel size mismatch");
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
    n_offset = THLongTensor_size(th_offset, 0);
    p_offset = THLongTensor_data(th_offset);
    if (THLongTensor_size(th_offset, 1) != D) {
      printf("DSCE ERROR: Offset size does not match.");
      return -1;
    }
  }

  // put exposed variable into _conv_foward;
  return _conv_tr_fw(p_in_feat, in_nchannel, p_out_feat, out_nchannel, p_kernel,
                     out_nrows, p_pixel_dist, p_stride, p_kernel_size,
                     p_dilation, region_type, p_offset, n_offset, D, m);
}

long convolution_backward(THFloatTensor *th_in_feat,
                          THFloatTensor *th_grad_in_feat,
                          THFloatTensor *th_grad_out_feat,
                          THFloatTensor *th_kernel,
                          THFloatTensor *th_grad_kernel,
                          THLongTensor *th_pixel_dist, THLongTensor *th_stride,
                          THLongTensor *th_kernel_size,
                          THLongTensor *th_dilation, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  long in_nrows, in_nchannel, _in_nchannel, out_nchannel, success,
      out_nrows = -1;
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
    printf("DSCE ERROR: Kernel channel size and input channel size mismatch");
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
                  p_pixel_dist, p_stride, p_kernel_size, p_dilation, D, m);
}

long convolution_transpose_backward(
    THFloatTensor *th_in_feat, THFloatTensor *th_grad_in_feat,
    THFloatTensor *th_grad_out_feat, THFloatTensor *th_kernel,
    THFloatTensor *th_grad_kernel, THLongTensor *th_pixel_dist,
    THLongTensor *th_stride, THLongTensor *th_kernel_size,
    THLongTensor *th_dilation, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  long in_nrows, in_nchannel, _in_nchannel, out_nchannel, success,
      out_nrows = -1;
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
    printf("DSCE ERROR: Kernel channel size and input channel size mismatch");
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
                     p_pixel_dist, p_stride, p_kernel_size, p_dilation, D, m);
}

long convolution_forward_gpu(THCudaTensor *th_in_feat,
                             THCudaTensor *th_out_feat, THCudaTensor *th_kernel,
                             THLongTensor *th_pixel_dist,
                             THLongTensor *th_stride,
                             THLongTensor *th_kernel_size,
                             THLongTensor *th_dilation, long region_type,
                             THLongTensor *th_offset, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  cudaStream_t stream = THCState_getCurrentStream(state);
  long in_nchannel, _in_nchannel, out_nchannel, success, out_nrows = -1;
  long n_offset = 0, *p_offset = NULL;
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
    printf("DSCE ERROR: Kernel channel size and input channel size mismatch");
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
    n_offset = THLongTensor_size(th_offset, 0);
    p_offset = THLongTensor_data(th_offset);
    if (THLongTensor_size(th_offset, 1) != D) {
      printf("DSCE ERROR: Offset size does not match.");
      return -1;
    }
  }

  // put exposed variable into _conv_foward;
  return _conv_fw_gpu(d_in_feat, in_nchannel, d_out_feat, out_nchannel,
                      d_kernel, out_nrows, p_pixel_dist, p_stride,
                      p_kernel_size, p_dilation, region_type, p_offset,
                      n_offset, stream, D, m);
}

long convolution_transpose_forward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_out_feat,
    THCudaTensor *th_kernel, THLongTensor *th_pixel_dist,
    THLongTensor *th_stride, THLongTensor *th_kernel_size,
    THLongTensor *th_dilation, long region_type, THLongTensor *th_offset,
    long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  cudaStream_t stream = THCState_getCurrentStream(state);
  long in_nchannel, _in_nchannel, out_nchannel, success, out_nrows = -1;
  long n_offset = 0, *p_offset = NULL;
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
    printf("DSCE ERROR: Kernel channel size and input channel size mismatch");
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
    n_offset = THLongTensor_size(th_offset, 0);
    p_offset = THLongTensor_data(th_offset);
    if (THLongTensor_size(th_offset, 1) != D) {
      printf("DSCE ERROR: Offset size does not match.");
      return -1;
    }
  }

  // put exposed variable into _conv_foward;
  return _conv_tr_fw_gpu(d_in_feat, in_nchannel, d_out_feat, out_nchannel,
                         d_kernel, out_nrows, p_pixel_dist, p_stride,
                         p_kernel_size, p_dilation, region_type, p_offset,
                         n_offset, stream, D, m);
}

long convolution_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_grad_out_feat, THCudaTensor *th_kernel,
    THCudaTensor *th_grad_kernel, THLongTensor *th_pixel_dist,
    THLongTensor *th_stride, THLongTensor *th_kernel_size,
    THLongTensor *th_dilation, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)
  cudaStream_t stream = THCState_getCurrentStream(state);
  long in_nrows, in_nchannel, _in_nchannel, out_nchannel, success,
      out_nrows = -1;
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
    printf("DSCE ERROR: Kernel channel size and input channel size mismatch");
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
                      p_pixel_dist, p_stride, p_kernel_size, p_dilation, stream,
                      D, m);
}

long convolution_transpose_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_grad_out_feat, THCudaTensor *th_kernel,
    THCudaTensor *th_grad_kernel, THLongTensor *th_pixel_dist,
    THLongTensor *th_stride, THLongTensor *th_kernel_size,
    THLongTensor *th_dilation, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)
  cudaStream_t stream = THCState_getCurrentStream(state);
  long in_nrows, in_nchannel, _in_nchannel, out_nchannel, success,
      out_nrows = -1;
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
    printf("DSCE ERROR: Kernel channel size and input channel size mismatch");
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
  return _conv_tr_bw_gpu(d_in_feat, d_grad_in_feat, in_nchannel,
                         d_grad_out_feat, out_nchannel, d_kernel, d_grad_kernel,
                         out_nrows, p_pixel_dist, p_stride, p_kernel_size,
                         p_dilation, stream, D, m);
}

long max_pooling_forward(THFloatTensor *th_in_feat, THFloatTensor *th_out_feat,
                         THLongTensor *th_mask_index,
                         THLongTensor *th_pixel_dist, THLongTensor *th_stride,
                         THLongTensor *th_kernel_size,
                         THLongTensor *th_dilation, long region_type,
                         THLongTensor *th_offset, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)
  long nchannel, success, out_nrows = -1;
  long n_offset = 0, *p_offset = NULL;
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
  THLongTensor_resize2d(th_mask_index, out_nrows, nchannel);

  // Pointers
  float *p_in_feat = THFloatTensor_data(th_in_feat);
  float *p_out_feat = THFloatTensor_data(th_out_feat);
  long *p_mask_index = THLongTensor_data(th_mask_index);

  // Custom Region Type
  if (region_type == 2) {
    n_offset = THLongTensor_size(th_offset, 0);
    p_offset = THLongTensor_data(th_offset);
    if (THLongTensor_size(th_offset, 1) != D) {
      printf("DSCE ERROR: Offset size does not match.");
      return -1;
    }
  }

  // put exposed variable into _conv_foward;
  return _max_pooling_fw(p_in_feat, p_out_feat, p_mask_index, nchannel,
                         out_nrows, p_pixel_dist, p_stride, p_kernel_size,
                         p_dilation, region_type, p_offset, n_offset, D, m);
}

long max_pooling_backward(THFloatTensor *th_in_feat,
                          THFloatTensor *th_grad_in_feat,
                          THFloatTensor *th_grad_out_feat,
                          THLongTensor *th_mask_index,
                          THLongTensor *th_pixel_dist, THLongTensor *th_stride,
                          THLongTensor *th_kernel_size,
                          THLongTensor *th_dilation, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)
  long in_nrows, nchannel, success, out_nrows = -1;
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
  long *p_mask_index = THLongTensor_data(th_mask_index);

  // put exposed variable into _conv_foward;
  return _max_pooling_bw(p_grad_in_feat, in_nrows, p_grad_out_feat, out_nrows,
                         p_mask_index, nchannel, p_pixel_dist, p_stride,
                         p_kernel_size, p_dilation, D, m);
}

long max_pooling_forward_gpu(THCudaTensor *th_in_feat,
                             THCudaTensor *th_out_feat,
                             THCudaLongTensor *th_mask_index,
                             THLongTensor *th_pixel_dist,
                             THLongTensor *th_stride,
                             THLongTensor *th_kernel_size,
                             THLongTensor *th_dilation, long region_type,
                             THLongTensor *th_offset, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  cudaStream_t stream = THCState_getCurrentStream(state);
  long nchannel, success, out_nrows = -1;
  long n_offset = 0, *p_offset = NULL;
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
  THCudaLongTensor_resize2d(state, th_mask_index, out_nrows, nchannel);

  // Pointers
  float *d_in_feat = THCudaTensor_data(state, th_in_feat);
  float *d_out_feat = THCudaTensor_data(state, th_out_feat);
  long *d_mask_index = THCudaLongTensor_data(state, th_mask_index);

  // Custom Region Type
  if (region_type == 2) {
    n_offset = THLongTensor_size(th_offset, 0);
    p_offset = THLongTensor_data(th_offset);
    if (THLongTensor_size(th_offset, 1) != D) {
      printf("DSCE ERROR: Offset size does not match.");
      return -1;
    }
  }

  // put exposed variable into _conv_foward;
  return _max_pooling_fw_gpu(d_in_feat, d_out_feat, out_nrows, d_mask_index,
                             nchannel, p_pixel_dist, p_stride, p_kernel_size,
                             p_dilation, region_type, p_offset, n_offset,
                             stream, D, m);
}

long max_pooling_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_grad_out_feat, THCudaLongTensor *th_mask_index,
    THLongTensor *th_pixel_dist, THLongTensor *th_stride,
    THLongTensor *th_kernel_size, THLongTensor *th_dilation, long D, void **m) {
  INIT_D_DIM_ARRY(th_pixel_dist, p_pixel_dist)
  INIT_D_DIM_ARRY(th_stride, p_stride)
  INIT_D_DIM_ARRY(th_kernel_size, p_kernel_size)
  INIT_D_DIM_ARRY(th_dilation, p_dilation)

  cudaStream_t stream = THCState_getCurrentStream(state);
  long in_nrows, nchannel, success, out_nrows = -1;
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
  long *d_mask_index = THCudaLongTensor_data(state, th_mask_index);

  // put exposed variable into _conv_foward;
  return _max_pooling_bw_gpu(d_grad_in_feat, in_nrows, d_grad_out_feat,
                             out_nrows, d_mask_index, nchannel, p_pixel_dist,
                             p_stride, p_kernel_size, p_dilation, stream, D, m);
}
