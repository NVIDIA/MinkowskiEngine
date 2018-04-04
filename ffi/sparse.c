#include <TH/TH.h>
#include <THC/THC.h>

#include <assert.h>
#include <stdbool.h>

#include "ffi/sparse_convolution.h"

#define CUDA_CHECK                                                             \
  error = cudaGetLastError();                                                  \
  if (error != cudaSuccess) {                                                  \
    printf("CUDA error: %s\n", cudaGetErrorString(error));                     \
    return -1;                                                                 \
  }

extern THCState *state;

long read_ffi_ptr(void **ptr) { return (long)(ptr[0]); }

void write_ffi_ptr(long p, void **ptr) { ptr[0] = (void *)p; }

long initialize_coords(THLongTensor *th_coords, long pixel_dist, long D,
                       void **m) {
  long *coords = THLongTensor_data(th_coords);
  int nrows = THLongTensor_size(th_coords, 0);
  int ncols = THLongTensor_size(th_coords, 1);
  assert(ncols - 1 == D);
  return _initialize_coords(coords, nrows, pixel_dist, D, m);
}

long initialize_coords_with_duplicates(THLongTensor *th_coords, long pixel_dist,
                                       long D, void **m) {
  long *coords = THLongTensor_data(th_coords);
  int nrows = THLongTensor_size(th_coords, 0);
  int ncols = THLongTensor_size(th_coords, 1);
  assert(ncols - 1 == D);
  return _initialize_coords_with_duplicates(coords, nrows, pixel_dist, D, m);
}

long get_coords(THLongTensor *th_coords, long pixel_dist, long D, void **m) {
  long success, nrows = -1;
  success = _get_num_coords(pixel_dist, &nrows, D, m);
  if (success < 0) {
    return success;
  }
  // Initialize torch and pass the pointer to fill out data
  THLongTensor_resize2d(th_coords, nrows, D + 1);
  THLongTensor_zero(th_coords);
  long *p_coords = THLongTensor_data(th_coords);
  success = _get_coords(p_coords, pixel_dist, D, m);
  return success;
}

long get_permutation(THLongTensor *th_permutation, long pixel_dist_src,
                     long pixel_dist_dst, long D, void **m) {
  long success, nrows = -1;
  success = _get_num_coords(pixel_dist_dst, &nrows, D, m);
  if (success < 0) {
    return success;
  }
  // Initialize torch and pass the pointer to fill out data
  THLongTensor_resize1d(th_permutation, nrows);
  THLongTensor_zero(th_permutation);
  long *p_permutation = THLongTensor_data(th_permutation);
  success =
      _get_permutation(p_permutation, pixel_dist_src, pixel_dist_dst, D, m);
  return success;
}

long get_index_map(THLongTensor *th_coords, THLongTensor *th_index_map,
                   long pixel_dist, long D, void **m) {
  long success, index_map_nrows = -1;
  success = _get_num_coords(pixel_dist, &index_map_nrows, D, m);
  if (success < 0) {
    return success;
  }

  // Set index map
  long nrows = THFloatTensor_size(th_coords, 0);
  THLongTensor_resize1d(th_index_map, nrows);
  THLongTensor_zero(th_index_map);
  long *p_coords = THLongTensor_data(th_coords);
  long *p_index_map = THLongTensor_data(th_index_map);

  success = _get_index_map(p_coords, nrows, p_index_map, index_map_nrows,
                           pixel_dist, D, m);
  return success;
}

long get_nrows(long pixel_dist, long D, void **m) {
  long success, nrows = -1;
  success = _get_num_coords(pixel_dist, &nrows, D, m);
  if (success < 0) {
    return success;
  } else {
    return nrows;
  }
}

void clear(long D, void **m) { _clear(D, m); }

long convolution_forward(THFloatTensor *th_in_feat, THFloatTensor *th_out_feat,
                         THFloatTensor *th_kernel, long pixel_dist, long stride,
                         long kernel_size, long dilation, long region_type,
                         THLongTensor *th_offset, long D, void **m) {
  // This will not take coords as the first initialization saved it in metadata
  // th_in_feat is 2D nrows, in_channel
  // th_kernel is 3D filter_volume, in_channel, out_channel
  long in_nrows, in_nchannel, _in_nchannel, out_nchannel, _out_nchannel,
      filter_volume, offset_size, success, out_nrows = -1;
  long n_offset, *p_offset;

  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  success = _initialize_out_coords(pixel_dist, stride, false, D, m);
  if (success < 0) {
    return success;
  }

  // Get the number of rows required to initialize the th_out_tensor
  success = _get_num_coords(pixel_dist * stride, &out_nrows, D, m);
  if (success < 0) {
    return success;
  }

  // expose all variables and resize out tensor
  filter_volume = THFloatTensor_size(th_kernel, 0);
  _in_nchannel = THFloatTensor_size(th_kernel, 1);
  out_nchannel = THFloatTensor_size(th_kernel, 2);
  in_nrows = THFloatTensor_size(th_in_feat, 0);
  in_nchannel = THFloatTensor_size(th_in_feat, 1);

  // Checks
  assert(_in_nchannel == in_nchannel);
  assert(_out_nchannel == out_nchannel);

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
                  out_nrows, pixel_dist, stride, kernel_size, dilation,
                  region_type, p_offset, n_offset, D, m);
}

long convolution_transpose_forward(THFloatTensor *th_in_feat,
                                   THFloatTensor *th_out_feat,
                                   THFloatTensor *th_kernel, long pixel_dist,
                                   long stride, long kernel_size, long dilation,
                                   long region_type, THLongTensor *th_offset,
                                   long D, void **m) {
  // This will not take coords as the first initialization saved it in metadata
  // th_in_feat is 2D nrows, in_channel
  // th_kernel is 3D filter_volume, in_channel, out_channel
  long in_nrows, in_nchannel, _in_nchannel, out_nchannel, _out_nchannel,
      filter_volume, offset_size, success, out_nrows = -1;
  long n_offset, *p_offset;

  if (pixel_dist % stride != 0) {
    printf("DSCE ERROR: pixel distance not divisible by stride.");
    return -1;
  }
  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  success = _initialize_out_coords(pixel_dist, stride, true, D, m);
  if (success < 0) {
    return success;
  }

  // Get the number of rows required to initialize the th_out_tensor
  success = _get_num_coords(pixel_dist / stride, &out_nrows, D, m);
  if (success < 0) {
    return success;
  }

  // expose all variables and resize out tensor
  filter_volume = THFloatTensor_size(th_kernel, 0);
  _in_nchannel = THFloatTensor_size(th_kernel, 1);
  out_nchannel = THFloatTensor_size(th_kernel, 2);
  in_nrows = THFloatTensor_size(th_in_feat, 0);
  in_nchannel = THFloatTensor_size(th_in_feat, 1);

  // Checks
  assert(_in_nchannel == in_nchannel);
  assert(_out_nchannel == out_nchannel);

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
                     out_nrows, pixel_dist, stride, kernel_size, dilation,
                     region_type, p_offset, n_offset, D, m);
}

long convolution_backward(THFloatTensor *th_in_feat,
                          THFloatTensor *th_grad_in_feat,
                          THFloatTensor *th_grad_out_feat,
                          THFloatTensor *th_kernel,
                          THFloatTensor *th_grad_kernel, long pixel_dist,
                          long stride, long kernel_size, long dilation, long D,
                          void **m) {
  // This will not take coords as the first initialization saved it in metadata
  // th_in_feat is 2D nrows, in_channel
  // th_kernel is 3D filter_volume, in_channel, out_channel
  long in_nrows, in_nchannel, _in_nchannel, out_nchannel, filter_volume,
      success, out_nrows = -1;

  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  success = _initialize_out_coords(pixel_dist, stride, false, D, m);
  if (success < 0) {
    return success;
  }

  // Get the number of rows required to initialize the th_out_tensor
  success = _get_num_coords(pixel_dist * stride, &out_nrows, D, m);
  if (success < 0) {
    return success;
  }

  // expose all variables and resize out tensor
  filter_volume = THFloatTensor_size(th_kernel, 0);
  _in_nchannel = THFloatTensor_size(th_kernel, 1);
  out_nchannel = THFloatTensor_size(th_kernel, 2);
  in_nrows = THFloatTensor_size(th_in_feat, 0);
  in_nchannel = THFloatTensor_size(th_in_feat, 1);

  // Checks
  assert(_in_nchannel == in_nchannel);
  // assert(pow(kernel_size, D) == filter_volume);

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
                  out_nchannel, p_kernel, p_grad_kernel, out_nrows, pixel_dist,
                  stride, kernel_size, dilation, D, m);
}

long convolution_transpose_backward(
    THFloatTensor *th_in_feat, THFloatTensor *th_grad_in_feat,
    THFloatTensor *th_grad_out_feat, THFloatTensor *th_kernel,
    THFloatTensor *th_grad_kernel, long pixel_dist, long stride,
    long kernel_size, long dilation, long D, void **m) {
  // This will not take coords as the first initialization saved it in metadata
  // th_in_feat is 2D nrows, in_channel
  // th_kernel is 3D filter_volume, in_channel, out_channel
  long in_nrows, in_nchannel, _in_nchannel, out_nchannel, filter_volume,
      success, out_nrows = -1;

  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  success = _initialize_out_coords(pixel_dist, stride, true, D, m);
  if (success < 0) {
    return success;
  }

  // Get the number of rows required to initialize the th_out_tensor
  success = _get_num_coords(pixel_dist / stride, &out_nrows, D, m);
  if (success < 0) {
    return success;
  }

  // expose all variables and resize out tensor
  filter_volume = THFloatTensor_size(th_kernel, 0);
  _in_nchannel = THFloatTensor_size(th_kernel, 1);
  out_nchannel = THFloatTensor_size(th_kernel, 2);
  in_nrows = THFloatTensor_size(th_in_feat, 0);
  in_nchannel = THFloatTensor_size(th_in_feat, 1);

  // Checks
  assert(_in_nchannel == in_nchannel);
  // assert(pow(kernel_size, D) == filter_volume);

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
                     pixel_dist, stride, kernel_size, dilation, D, m);
}

long convolution_forward_gpu(THCudaTensor *th_in_feat,
                             THCudaTensor *th_out_feat, THCudaTensor *th_kernel,
                             long pixel_dist, long stride, long kernel_size,
                             long dilation, long region_type,
                             THLongTensor *th_offset, long D, void **m) {
  // This will not take coords as the first initialization saved it in metadata
  // th_in_feat is 2D nrows, in_channel
  // th_kernel is 3D filter_volume, in_channel, out_channel
  cudaStream_t stream = THCState_getCurrentStream(state);
  long in_nrows, in_nchannel, _in_nchannel, out_nchannel, _out_nchannel,
      filter_volume, success, out_nrows = -1;
  long n_offset, *p_offset;
  cudaError_t error;

  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  success = _initialize_out_coords(pixel_dist, stride, false, D, m);
  if (success < 0) {
    return success;
  }

  // Get the number of rows required to initialize the th_out_tensor
  success = _get_num_coords(pixel_dist * stride, &out_nrows, D, m);
  if (success < 0) {
    return success;
  }

  // expose all variables and resize out tensor
  filter_volume = THCudaTensor_size(state, th_kernel, 0);
  _in_nchannel = THCudaTensor_size(state, th_kernel, 1);
  out_nchannel = THCudaTensor_size(state, th_kernel, 2);
  in_nrows = THCudaTensor_size(state, th_in_feat, 0);
  in_nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Checks
  assert(_in_nchannel == in_nchannel);
  assert(_out_nchannel == out_nchannel);

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
                      d_kernel, out_nrows, pixel_dist, stride, kernel_size,
                      dilation, region_type, p_offset, n_offset, stream, D, m);
}

long convolution_transpose_forward_gpu(THCudaTensor *th_in_feat,
                                       THCudaTensor *th_out_feat,
                                       THCudaTensor *th_kernel, long pixel_dist,
                                       long stride, long kernel_size,
                                       long dilation, long region_type,
                                       THLongTensor *th_offset, long D,
                                       void **m) {
  // This will not take coords as the first initialization saved it in metadata
  // th_in_feat is 2D nrows, in_channel
  // th_kernel is 3D filter_volume, in_channel, out_channel
  cudaStream_t stream = THCState_getCurrentStream(state);
  long in_nrows, in_nchannel, _in_nchannel, out_nchannel, _out_nchannel,
      filter_volume, success, out_nrows = -1;
  long n_offset, *p_offset;
  cudaError_t error;

  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  success = _initialize_out_coords(pixel_dist, stride, true, D, m);
  if (success < 0) {
    return success;
  }

  // Get the number of rows required to initialize the th_out_tensor
  success = _get_num_coords(pixel_dist / stride, &out_nrows, D, m);
  if (success < 0) {
    return success;
  }

  // expose all variables and resize out tensor
  filter_volume = THCudaTensor_size(state, th_kernel, 0);
  _in_nchannel = THCudaTensor_size(state, th_kernel, 1);
  out_nchannel = THCudaTensor_size(state, th_kernel, 2);
  in_nrows = THCudaTensor_size(state, th_in_feat, 0);
  in_nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Checks
  assert(_in_nchannel == in_nchannel);
  assert(_out_nchannel == out_nchannel);

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
                         d_kernel, out_nrows, pixel_dist, stride, kernel_size,
                         dilation, region_type, p_offset, n_offset, stream, D,
                         m);
}

long convolution_backward_gpu(THCudaTensor *th_in_feat,
                              THCudaTensor *th_grad_in_feat,
                              THCudaTensor *th_grad_out_feat,
                              THCudaTensor *th_kernel,
                              THCudaTensor *th_grad_kernel, long pixel_dist,
                              long stride, long kernel_size, long dilation,
                              long D, void **m) {
  // This will not take coords as the first initialization saved it in metadata
  // th_in_feat is 2D nrows, in_channel
  // th_kernel is 3D filter_volume, in_channel, out_channel
  cudaStream_t stream = THCState_getCurrentStream(state);
  long in_nrows, in_nchannel, _in_nchannel, out_nchannel, filter_volume,
      success, out_nrows = -1;
  cudaError_t error;

  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  success = _initialize_out_coords(pixel_dist, stride, false, D, m);
  if (success < 0) {
    return success;
  }

  // Get the number of rows required to initialize the th_out_tensor
  success = _get_num_coords(pixel_dist * stride, &out_nrows, D, m);
  if (success < 0) {
    return success;
  }

  // expose all variables and resize out tensor
  filter_volume = THCudaTensor_size(state, th_kernel, 0);
  _in_nchannel = THCudaTensor_size(state, th_kernel, 1);
  out_nchannel = THCudaTensor_size(state, th_kernel, 2);
  in_nrows = THCudaTensor_size(state, th_in_feat, 0);
  in_nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Checks
  assert(_in_nchannel == in_nchannel);

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
                      pixel_dist, stride, kernel_size, dilation, stream, D, m);
}

long convolution_transpose_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_grad_out_feat, THCudaTensor *th_kernel,
    THCudaTensor *th_grad_kernel, long pixel_dist, long stride,
    long kernel_size, long dilation, long D, void **m) {
  // This will not take coords as the first initialization saved it in metadata
  // th_in_feat is 2D nrows, in_channel
  // th_kernel is 3D filter_volume, in_channel, out_channel
  cudaStream_t stream = THCState_getCurrentStream(state);
  long in_nrows, in_nchannel, _in_nchannel, out_nchannel, filter_volume,
      success, out_nrows = -1;
  cudaError_t error;

  // Check if the input pixel dist map exists. Output map will be generate
  // automatically inside the convolution kernel.
  success = _initialize_out_coords(pixel_dist, stride, true, D, m);
  if (success < 0) {
    return success;
  }

  // Get the number of rows required to initialize the th_out_tensor
  success = _get_num_coords(pixel_dist / stride, &out_nrows, D, m);
  if (success < 0) {
    return success;
  }

  // expose all variables and resize out tensor
  filter_volume = THCudaTensor_size(state, th_kernel, 0);
  _in_nchannel = THCudaTensor_size(state, th_kernel, 1);
  out_nchannel = THCudaTensor_size(state, th_kernel, 2);
  in_nrows = THCudaTensor_size(state, th_in_feat, 0);
  in_nchannel = THCudaTensor_size(state, th_in_feat, 1);

  // Checks
  assert(_in_nchannel == in_nchannel);

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
                         out_nrows, pixel_dist, stride, kernel_size, dilation,
                         stream, D, m);
}
