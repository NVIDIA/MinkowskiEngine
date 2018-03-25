// All functions exposed to python
long read_ffi_ptr(void **ptr);
void write_ffi_ptr(long p, void **ptr);

// Initializations
long initialize_coords(THLongTensor *th_coords, long pixel_dist, long D,
                       void **m);

long initialize_coords_with_duplicates(THLongTensor *th_coords, long pixel_dist,
                                       long D, void **m);

long get_coords(THLongTensor *th_coords, long pixel_dist, long D, void **m);

long get_nrows(long pixel_dist, long D, void **m);

long get_permutation(THLongTensor *th_permutation, long pixel_dist_src,
                     long pixel_dist_dst, long D, void **m);

long get_index_map(THLongTensor *th_coords, THLongTensor *th_index_map,
                   long pixel_dist, long D, void **m);

// Clear
void clear(long D, void **m);

// Convolutions
long convolution_forward(THFloatTensor *th_in_feat, THFloatTensor *th_out_feat,
                         THFloatTensor *th_kernel, THFloatTensor *th_bias,
                         long pixel_dist, long stride, long kernel_size,
                         long dilation, long region_type,
                         THLongTensor *th_neighbor, long D, void **m);

long convolution_transpose_forward(THFloatTensor *th_in_feat,
                                   THFloatTensor *th_out_feat,
                                   THFloatTensor *th_kernel,
                                   THFloatTensor *th_bias, long pixel_dist,
                                   long stride, long kernel_size, long dilation,
                                   long region_type, THLongTensor *th_neighbor,
                                   long D, void **m);

long convolution_backward(
    THFloatTensor *th_in_feat, THFloatTensor *th_grad_in_feat,
    THFloatTensor *th_grad_out_feat, THFloatTensor *th_kernel,
    THFloatTensor *th_grad_kernel, THFloatTensor *th_grad_bias, long pixel_dist,
    long stride, long kernel_size, long dilation, long D, void **m);

long convolution_transpose_backward(
    THFloatTensor *th_in_feat, THFloatTensor *th_grad_in_feat,
    THFloatTensor *th_grad_out_feat, THFloatTensor *th_kernel,
    THFloatTensor *th_grad_kernel, THFloatTensor *th_grad_bias, long pixel_dist,
    long stride, long kernel_size, long dilation, long D, void **m);

long convolution_forward_gpu(THCudaTensor *th_in_feat,
                             THCudaTensor *th_out_feat, THCudaTensor *th_kernel,
                             THCudaTensor *th_bias, long pixel_dist,
                             long stride, long kernel_size, long dilation,
                             long region_type, THLongTensor *th_neighbor,
                             long D, void **m);

long convolution_transpose_forward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_out_feat,
    THCudaTensor *th_kernel, THCudaTensor *th_bias, long pixel_dist,
    long stride, long kernel_size, long dilation, long region_type,
    THLongTensor *th_neighbor, long D, void **m);

long convolution_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_grad_out_feat, THCudaTensor *th_kernel,
    THCudaTensor *th_grad_kernel, THCudaTensor *th_grad_bias, long pixel_dist,
    long stride, long kernel_size, long dilation, long D, void **m);

long convolution_transpose_backward_gpu(
    THCudaTensor *th_in_feat, THCudaTensor *th_grad_in_feat,
    THCudaTensor *th_grad_out_feat, THCudaTensor *th_kernel,
    THCudaTensor *th_grad_kernel, THCudaTensor *th_grad_bias, long pixel_dist,
    long stride, long kernel_size, long dilation, long D, void **m);
