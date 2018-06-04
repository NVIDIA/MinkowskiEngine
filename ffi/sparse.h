// All functions exposed to python
long read_ffi_ptr(void **ptr);
void write_ffi_ptr(long p, void **ptr);

// Initializations
long initialize_coords(THIntTensor *th_coords, THIntTensor *th_pixel_dist,
                       long D, void **m);

long initialize_coords_with_duplicates(THIntTensor *th_coords,
                                       THIntTensor *th_pixel_dist, long D,
                                       void **m);

long get_coords(THIntTensor *th_coords, THIntTensor *th_pixel_dist, long D,
                void **m);

long get_nrows(THIntTensor *th_pixel_dist, long D, void **m);

long get_permutation(THIntTensor *th_permutation,
                     THIntTensor *th_pixel_dist_src,
                     THIntTensor *th_pixel_dist_dst, long D, void **m);

long get_index_map(THIntTensor *th_coords, THIntTensor *th_index_map,
                   THIntTensor *th_pixel_dist, long D, void **m);

// Clear
void clear(long D, void **m);

// Convolutions
long convolution_forward(THFloatTensor *th_in_feat, THFloatTensor *th_out_feat,
                         THFloatTensor *th_kernel, THIntTensor *th_pixel_dist,
                         THIntTensor *th_stride, THIntTensor *th_kernel_size,
                         THIntTensor *th_dilation, long region_type,
                         THIntTensor *th_neighbor, long D, void **m);

long convolution_transpose_forward(THFloatTensor *th_in_feat,
                                   THFloatTensor *th_out_feat,
                                   THFloatTensor *th_kernel,
                                   THIntTensor *th_pixel_dist,
                                   THIntTensor *th_stride,
                                   THIntTensor *th_kernel_size,
                                   THIntTensor *th_dilation, long region_type,
                                   THIntTensor *th_neighbor, long D, void **m);

long convolution_backward(THFloatTensor *th_in_feat,
                          THFloatTensor *th_grad_in_feat,
                          THFloatTensor *th_grad_out_feat,
                          THFloatTensor *th_kernel,
                          THFloatTensor *th_grad_kernel,
                          THIntTensor *th_pixel_dist, THIntTensor *th_stride,
                          THIntTensor *th_kernel_size, THIntTensor *th_dilation,
                          long D, void **m);

long convolution_transpose_backward(
    THFloatTensor *th_in_feat, THFloatTensor *th_grad_in_feat,
    THFloatTensor *th_grad_out_feat, THFloatTensor *th_kernel,
    THFloatTensor *th_grad_kernel, THIntTensor *th_pixel_dist,
    THIntTensor *th_stride, THIntTensor *th_kernel_size,
    THIntTensor *th_dilation, long D, void **m);

long convolution_forward_gpu(THCudaTensor *th_in_feat,
                             THCudaTensor *th_out_feat, THCudaTensor *th_kernel,
                             THIntTensor *th_pixel_dist, THIntTensor *th_stride,
                             THIntTensor *th_kernel_size,
                             THIntTensor *th_dilation, long region_type,
                             THIntTensor *th_neighbor, long D, void **m);

long convolution_transpose_forward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_out_feat,
    THCudaTensor *th_kernel, THIntTensor *th_pixel_dist, THIntTensor *th_stride,
    THIntTensor *th_kernel_size, THIntTensor *th_dilation, long region_type,
    THIntTensor *th_neighbor, long D, void **m);

long convolution_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_grad_out_feat, THCudaTensor *th_kernel,
    THCudaTensor *th_grad_kernel, THIntTensor *th_pixel_dist,
    THIntTensor *th_stride, THIntTensor *th_kernel_size,
    THIntTensor *th_dilation, long D, void **m);

long convolution_transpose_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_grad_out_feat, THCudaTensor *th_kernel,
    THCudaTensor *th_grad_kernel, THIntTensor *th_pixel_dist,
    THIntTensor *th_stride, THIntTensor *th_kernel_size,
    THIntTensor *th_dilation, long D, void **m);

long max_pooling_forward(THFloatTensor *th_in_feat, THFloatTensor *th_out_feat,
                         THIntTensor *th_mask_index, THIntTensor *th_pixel_dist,
                         THIntTensor *th_stride, THIntTensor *th_kernel_size,
                         THIntTensor *th_dilation, long region_type,
                         THIntTensor *th_offset, long D, void **m);

long max_pooling_backward(THFloatTensor *th_in_feat,
                          THFloatTensor *th_grad_in_feat,
                          THFloatTensor *th_grad_out_feat,
                          THIntTensor *th_mask_index,
                          THIntTensor *th_pixel_dist, THIntTensor *th_stride,
                          THIntTensor *th_kernel_size, THIntTensor *th_dilation,
                          long D, void **m);

long max_pooling_forward_gpu(THCudaTensor *th_in_feat,
                             THCudaTensor *th_out_feat,
                             THCudaIntTensor *th_mask_index,
                             THIntTensor *th_pixel_dist, THIntTensor *th_stride,
                             THIntTensor *th_kernel_size,
                             THIntTensor *th_dilation, long region_type,
                             THIntTensor *th_offset, long D, void **m);

long max_pooling_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_grad_out_feat, THCudaIntTensor *th_mask_index,
    THIntTensor *th_pixel_dist, THIntTensor *th_stride,
    THIntTensor *th_kernel_size, THIntTensor *th_dilation, long D, void **m);

long nonzero_avg_pooling_forward(THFloatTensor *th_in_feat,
                                 THFloatTensor *th_out_feat,
                                 THIntTensor *th_num_nonzero,
                                 THIntTensor *th_pixel_dist,
                                 THIntTensor *th_stride,
                                 THIntTensor *th_kernel_size,
                                 THIntTensor *th_dilation, long region_type,
                                 THIntTensor *th_offset, long D, void **m);

long nonzero_avg_pooling_backward(
    THFloatTensor *th_in_feat, THFloatTensor *th_grad_in_feat,
    THFloatTensor *th_grad_out_feat, THIntTensor *th_num_nonzero,
    THIntTensor *th_pixel_dist, THIntTensor *th_stride,
    THIntTensor *th_kernel_size, THIntTensor *th_dilation, long D, void **m);

long nonzero_avg_pooling_forward_gpu(THCudaTensor *th_in_feat,
                                     THCudaTensor *th_out_feat,
                                     THCudaIntTensor *th_num_nonzero,
                                     THIntTensor *th_pixel_dist,
                                     THIntTensor *th_stride,
                                     THIntTensor *th_kernel_size,
                                     THIntTensor *th_dilation, long region_type,
                                     THIntTensor *th_offset, long D, void **m);

long nonzero_avg_pooling_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_grad_out_feat, THCudaIntTensor *th_num_nonzero,
    THIntTensor *th_pixel_dist, THIntTensor *th_stride,
    THIntTensor *th_kernel_size, THIntTensor *th_dilation, long D, void **m);

long global_avg_pooling_forward(THFloatTensor *th_in_feat,
                                THFloatTensor *th_out_feat,
                                THIntTensor *th_num_nonzero,
                                THIntTensor *th_pixel_dist, long batch_size,
                                long D, void **m);

long global_avg_pooling_backward(THFloatTensor *th_in_feat,
                                 THFloatTensor *th_grad_in_feat,
                                 THFloatTensor *th_grad_out_feat,
                                 THIntTensor *th_num_nonzero,
                                 THIntTensor *th_pixel_dist, long D, void **m);

long global_avg_pooling_forward_gpu(THCudaTensor *th_in_feat,
                                    THCudaTensor *th_out_feat,
                                    THCudaIntTensor *th_num_nonzero,
                                    THIntTensor *th_pixel_dist, long batch_size,
                                    long D, void **m);

long global_avg_pooling_backward_gpu(THCudaTensor *th_in_feat,
                                     THCudaTensor *th_grad_in_feat,
                                     THCudaTensor *th_grad_out_feat,
                                     THCudaIntTensor *th_num_nonzero,
                                     THIntTensor *th_pixel_dist, long D,
                                     void **m);

long global_broadcast_forward(THFloatTensor *th_in_feat,
                              THFloatTensor *th_in_feat_global,
                              THFloatTensor *th_out_feat,
                              THIntTensor *th_pixel_dist, long op, long D,
                              void **m);

long global_broadcast_backward(THFloatTensor *th_in_feat,
                               THFloatTensor *th_grad_in_feat,
                               THFloatTensor *th_in_feat_global,
                               THFloatTensor *th_grad_in_feat_global,
                               THFloatTensor *th_grad_out_feat,
                               THIntTensor *th_pixel_dist, long op, long D,
                               void **m);

long global_broadcast_forward_gpu(THCudaTensor *th_in_feat,
                                  THCudaTensor *th_in_feat_global,
                                  THCudaTensor *th_out_feat,
                                  THIntTensor *th_pixel_dist, long op, long D,
                                  void **m);

long global_broadcast_backward_gpu(THCudaTensor *th_in_feat,
                                   THCudaTensor *th_grad_in_feat,
                                   THCudaTensor *th_in_feat_global,
                                   THCudaTensor *th_grad_in_feat_global,
                                   THCudaTensor *th_grad_out_feat,
                                   THIntTensor *th_pixel_dist, long op, long D,
                                   void **m);
