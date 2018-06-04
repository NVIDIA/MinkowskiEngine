// All functions exposed to python
int read_ffi_ptr(void **ptr);
void write_ffi_ptr(int p, void **ptr);

// Initializations
int initialize_coords(THIntTensor *th_coords, THIntTensor *th_pixel_dist, int D,
                      void **m);

int initialize_coords_with_duplicates(THIntTensor *th_coords,
                                      THIntTensor *th_pixel_dist, int D,
                                      void **m);

int get_coords(THIntTensor *th_coords, THIntTensor *th_pixel_dist, int D,
               void **m);

int get_nrows(THIntTensor *th_pixel_dist, int D, void **m);

int get_permutation(THIntTensor *th_permutation, THIntTensor *th_pixel_dist_src,
                    THIntTensor *th_pixel_dist_dst, int D, void **m);

int get_index_map(THIntTensor *th_coords, THIntTensor *th_index_map,
                  THIntTensor *th_pixel_dist, int D, void **m);

// Clear
void clear(int D, void **m);

// Convolutions
int convolution_forward(THFloatTensor *th_in_feat, THFloatTensor *th_out_feat,
                        THFloatTensor *th_kernel, THIntTensor *th_pixel_dist,
                        THIntTensor *th_stride, THIntTensor *th_kernel_size,
                        THIntTensor *th_dilation, int region_type,
                        THIntTensor *th_neighbor, int D, void **m);

int convolution_transpose_forward(THFloatTensor *th_in_feat,
                                  THFloatTensor *th_out_feat,
                                  THFloatTensor *th_kernel,
                                  THIntTensor *th_pixel_dist,
                                  THIntTensor *th_stride,
                                  THIntTensor *th_kernel_size,
                                  THIntTensor *th_dilation, int region_type,
                                  THIntTensor *th_neighbor, int D, void **m);

int convolution_backward(THFloatTensor *th_in_feat,
                         THFloatTensor *th_grad_in_feat,
                         THFloatTensor *th_grad_out_feat,
                         THFloatTensor *th_kernel,
                         THFloatTensor *th_grad_kernel,
                         THIntTensor *th_pixel_dist, THIntTensor *th_stride,
                         THIntTensor *th_kernel_size, THIntTensor *th_dilation,
                         int D, void **m);

int convolution_transpose_backward(
    THFloatTensor *th_in_feat, THFloatTensor *th_grad_in_feat,
    THFloatTensor *th_grad_out_feat, THFloatTensor *th_kernel,
    THFloatTensor *th_grad_kernel, THIntTensor *th_pixel_dist,
    THIntTensor *th_stride, THIntTensor *th_kernel_size,
    THIntTensor *th_dilation, int D, void **m);

int convolution_forward_gpu(THCudaTensor *th_in_feat, THCudaTensor *th_out_feat,
                            THCudaTensor *th_kernel, THIntTensor *th_pixel_dist,
                            THIntTensor *th_stride, THIntTensor *th_kernel_size,
                            THIntTensor *th_dilation, int region_type,
                            THIntTensor *th_neighbor, int D, void **m);

int convolution_transpose_forward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_out_feat,
    THCudaTensor *th_kernel, THIntTensor *th_pixel_dist, THIntTensor *th_stride,
    THIntTensor *th_kernel_size, THIntTensor *th_dilation, int region_type,
    THIntTensor *th_neighbor, int D, void **m);

int convolution_backward_gpu(THCudaTensor *th_in_feat,
                             THCudaTensor *th_grad_in_feat,
                             THCudaTensor *th_grad_out_feat,
                             THCudaTensor *th_kernel,
                             THCudaTensor *th_grad_kernel,
                             THIntTensor *th_pixel_dist, THIntTensor *th_stride,
                             THIntTensor *th_kernel_size,
                             THIntTensor *th_dilation, int D, void **m);

int convolution_transpose_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_grad_out_feat, THCudaTensor *th_kernel,
    THCudaTensor *th_grad_kernel, THIntTensor *th_pixel_dist,
    THIntTensor *th_stride, THIntTensor *th_kernel_size,
    THIntTensor *th_dilation, int D, void **m);

int max_pooling_forward(THFloatTensor *th_in_feat, THFloatTensor *th_out_feat,
                        THIntTensor *th_mask_index, THIntTensor *th_pixel_dist,
                        THIntTensor *th_stride, THIntTensor *th_kernel_size,
                        THIntTensor *th_dilation, int region_type,
                        THIntTensor *th_offset, int D, void **m);

int max_pooling_backward(THFloatTensor *th_in_feat,
                         THFloatTensor *th_grad_in_feat,
                         THFloatTensor *th_grad_out_feat,
                         THIntTensor *th_mask_index, THIntTensor *th_pixel_dist,
                         THIntTensor *th_stride, THIntTensor *th_kernel_size,
                         THIntTensor *th_dilation, int D, void **m);

int max_pooling_forward_gpu(THCudaTensor *th_in_feat, THCudaTensor *th_out_feat,
                            THCudaIntTensor *th_mask_index,
                            THIntTensor *th_pixel_dist, THIntTensor *th_stride,
                            THIntTensor *th_kernel_size,
                            THIntTensor *th_dilation, int region_type,
                            THIntTensor *th_offset, int D, void **m);

int max_pooling_backward_gpu(THCudaTensor *th_in_feat,
                             THCudaTensor *th_grad_in_feat,
                             THCudaTensor *th_grad_out_feat,
                             THCudaIntTensor *th_mask_index,
                             THIntTensor *th_pixel_dist, THIntTensor *th_stride,
                             THIntTensor *th_kernel_size,
                             THIntTensor *th_dilation, int D, void **m);

int nonzero_avg_pooling_forward(THFloatTensor *th_in_feat,
                                THFloatTensor *th_out_feat,
                                THIntTensor *th_num_nonzero,
                                THIntTensor *th_pixel_dist,
                                THIntTensor *th_stride,
                                THIntTensor *th_kernel_size,
                                THIntTensor *th_dilation, int region_type,
                                THIntTensor *th_offset, int D, void **m);

int nonzero_avg_pooling_backward(
    THFloatTensor *th_in_feat, THFloatTensor *th_grad_in_feat,
    THFloatTensor *th_grad_out_feat, THIntTensor *th_num_nonzero,
    THIntTensor *th_pixel_dist, THIntTensor *th_stride,
    THIntTensor *th_kernel_size, THIntTensor *th_dilation, int D, void **m);

int nonzero_avg_pooling_forward_gpu(THCudaTensor *th_in_feat,
                                    THCudaTensor *th_out_feat,
                                    THCudaIntTensor *th_num_nonzero,
                                    THIntTensor *th_pixel_dist,
                                    THIntTensor *th_stride,
                                    THIntTensor *th_kernel_size,
                                    THIntTensor *th_dilation, int region_type,
                                    THIntTensor *th_offset, int D, void **m);

int nonzero_avg_pooling_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_grad_out_feat, THCudaIntTensor *th_num_nonzero,
    THIntTensor *th_pixel_dist, THIntTensor *th_stride,
    THIntTensor *th_kernel_size, THIntTensor *th_dilation, int D, void **m);

int global_avg_pooling_forward(THFloatTensor *th_in_feat,
                               THFloatTensor *th_out_feat,
                               THIntTensor *th_num_nonzero,
                               THIntTensor *th_pixel_dist, int batch_size,
                               int D, void **m);

int global_avg_pooling_backward(THFloatTensor *th_in_feat,
                                THFloatTensor *th_grad_in_feat,
                                THFloatTensor *th_grad_out_feat,
                                THIntTensor *th_num_nonzero,
                                THIntTensor *th_pixel_dist, int D, void **m);

int global_avg_pooling_forward_gpu(THCudaTensor *th_in_feat,
                                   THCudaTensor *th_out_feat,
                                   THCudaIntTensor *th_num_nonzero,
                                   THIntTensor *th_pixel_dist, int batch_size,
                                   int D, void **m);

int global_avg_pooling_backward_gpu(THCudaTensor *th_in_feat,
                                    THCudaTensor *th_grad_in_feat,
                                    THCudaTensor *th_grad_out_feat,
                                    THCudaIntTensor *th_num_nonzero,
                                    THIntTensor *th_pixel_dist, int D,
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
