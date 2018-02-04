// All CPP functions that will be used in sparse.c
#include <cuda_runtime.h>

extern long _initialize_coords(long *coords, long nrows, long pixel_dist,
                               long D, void **metadata);

extern long _initialize_out_coords(long pixel_dist, long stride, long D,
                                   void **metadata);

extern long _initialize_coords_with_duplicates(long *coords, long nrows,
                                               long pixel_dist, long D,
                                               void **metadata);

extern long _get_index_map(const long *coords, long nrows, long *p_index_map,
                           long index_map_nrows, long pixel_dist, long D,
                           void **metadata);

extern long _get_coords(long *coords, long pixel_dist, long D, void **metadata);

extern long _get_num_coords(long pixel_dist, long *p_nrows, long D,
                            void **metadata);

extern long _get_permutation(long *p_permutation, long pixel_dist_src,
                             long pixel_dist_dst, long D, void **metadata);

extern void _clear(long D, void **metadata);

extern long _conv_fw(float *p_in_feat, long in_nchannel, float *p_out_feat,
                     long out_nchannel, float *p_kernel, float *p_bias,
                     long out_nrows, long pixel_dist, long stride,
                     long kernel_size, long dilation, long region_type,
                     long *p_offset, long n_offset, long D, void **metadata);

extern long _conv_bw(float *p_in_feat, float *p_grad_in_feat, long in_nchannel,
                     float *p_grad_out_feat, long out_nchannel, float *p_kernel,
                     float *p_grad_kernel, float *p_grad_bias, long out_nrows,
                     long pixel_dist, long stride, long kernel_size,
                     long dilation, long D, void **metadata);

extern long _conv_fw_gpu(float *d_in_feat, long in_nchannel, float *d_out_feat,
                         long out_nchannel, float *d_kernel, float *d_bias,
                         long out_nrows, long pixel_dist, long stride,
                         long kernel_size, long dilation, long region_type,
                         long *p_offset, long n_offset, cudaStream_t stream,
                         long D, void **metadata);

extern long _conv_bw_gpu(float *d_in_feat, float *d_grad_in_feat,
                         long in_nchannel, float *d_grad_out_feat,
                         long out_nchannel, float *d_kernel,
                         float *d_grad_kernel, float *d_grad_bias,
                         long out_nrows, long pixel_dist, long stride,
                         long kernel_size, long dilation, cudaStream_t stream,
                         long D, void **metadata);
