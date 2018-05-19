// All CPP functions that will be used in sparse.c
#include <cuda_runtime.h>

extern long _initialize_coords(long *coords, long nrows, long *p_pixel_dist,
                               long D, void **metadata);

extern long _initialize_out_coords(long *p_pixel_dist, long *p_stride,
                                   bool is_transpose, long D, void **metadata);

extern long _initialize_coords_with_duplicates(long *coords, long nrows,
                                               long *p_pixel_dist, long D,
                                               void **metadata);

extern long _get_index_map(const long *coords, long nrows, long *p_index_map,
                           long index_map_nrows, long *p_pixel_dist, long D,
                           void **metadata);

extern long _get_coords(long *coords, long *p_pixel_dist, long D,
                        void **metadata);

extern long _get_num_coords(long *p_pixel_dist, long *p_nrows, long D,
                            void **metadata);

extern long _get_permutation(long *p_permutation, long *p_pixel_dist_src,
                             long *p_pixel_dist_dst, long D, void **metadata);

extern void _clear(long D, void **metadata);

extern long _conv_fw(float *p_in_feat, long in_nchannel, float *p_out_feat,
                     long out_nchannel, float *p_kernel, long out_nrows,
                     long *p_pixel_dist, long *p_stride, long *p_kernel_size,
                     long *p_dilation, long region_type, long *p_offset,
                     long n_offset, long D, void **metadata);

extern long _conv_tr_fw(float *p_in_feat, long in_nchannel, float *p_out_feat,
                        long out_nchannel, float *p_kernel, long out_nrows,
                        long *p_pixel_dist, long *p_stride, long *p_kernel_size,
                        long *p_dilation, long region_type, long *p_offset,
                        long n_offset, long D, void **metadata);

extern long _conv_bw(float *p_in_feat, float *p_grad_in_feat, long in_nchannel,
                     float *p_grad_out_feat, long out_nchannel, float *p_kernel,
                     float *p_grad_kernel, long out_nrows, long *p_pixel_dist,
                     long *p_stride, long *p_kernel_size, long *p_dilation,
                     long D, void **metadata);

extern long _conv_tr_bw(float *p_in_feat, float *p_grad_in_feat,
                        long in_nchannel, float *p_grad_out_feat,
                        long out_nchannel, float *p_kernel,
                        float *p_grad_kernel, long out_nrows,
                        long *p_pixel_dist, long *p_stride, long *p_kernel_size,
                        long *p_dilation, long D, void **metadata);

extern long _conv_fw_gpu(float *d_in_feat, long in_nchannel, float *d_out_feat,
                         long out_nchannel, float *d_kernel, long out_nrows,
                         long *p_pixel_dist, long *p_stride,
                         long *p_kernel_size, long *p_dilation,
                         long region_type, long *p_offset, long n_offset,
                         cudaStream_t stream, long D, void **metadata);

extern long _conv_tr_fw_gpu(float *d_in_feat, long in_nchannel,
                            float *d_out_feat, long out_nchannel,
                            float *d_kernel, long out_nrows, long *p_pixel_dist,
                            long *p_stride, long *p_kernel_size,
                            long *p_dilation, long region_type, long *p_offset,
                            long n_offset, cudaStream_t stream, long D,
                            void **metadata);

extern long _conv_bw_gpu(float *d_in_feat, float *d_grad_in_feat,
                         long in_nchannel, float *d_grad_out_feat,
                         long out_nchannel, float *d_kernel,
                         float *d_grad_kernel, long out_nrows,
                         long *p_pixel_dist, long *p_stride,
                         long *p_kernel_size, long *p_dilation,
                         cudaStream_t stream, long D, void **metadata);

extern long _conv_tr_bw_gpu(float *d_in_feat, float *d_grad_in_feat,
                            long in_nchannel, float *d_grad_out_feat,
                            long out_nchannel, float *d_kernel,
                            float *d_grad_kernel, long out_nrows,
                            long *p_pixel_dist, long *p_stride,
                            long *p_kernel_size, long *p_dilation,
                            cudaStream_t stream, long D, void **metadata);

extern long _max_pooling_fw(float *p_in_feat, float *p_out_feat,
                            int64_t *p_mask_index, int64_t nchannel,
                            int64_t out_nrows, int64_t *p_pixel_dist,
                            int64_t *p_stride, int64_t *p_kernel_size,
                            int64_t *p_dilation, int64_t region_type,
                            int64_t *p_offset, int64_t n_offset, long D,
                            void **metadata);

extern long _max_pooling_bw(float *p_grad_in_feat, int64_t in_nrows,
                            float *p_grad_out_feat, int64_t out_nrows,
                            int64_t *p_mask_index, int64_t nchannel,
                            int64_t *p_pixel_dist, int64_t *p_stride,
                            int64_t *p_kernel_size, int64_t *p_dilation, long D,
                            void **metadata);

extern long _max_pooling_fw_gpu(float *d_in_feat, float *d_out_feat,
                                int64_t out_nrows, int64_t *d_mask_index,
                                int64_t nchannel, int64_t *p_pixel_dist,
                                int64_t *p_stride, int64_t *p_kernel_size,
                                int64_t *p_dilation, int64_t region_type,
                                int64_t *p_offset, int64_t n_offset,
                                cudaStream_t stream, long D, void **metadata);

extern long _max_pooling_bw_gpu(float *d_grad_in_feat, int64_t in_nrows,
                                float *d_grad_out_feat, int64_t out_nrows,
                                int64_t *d_mask_index, int64_t nchannel,
                                int64_t *p_pixel_dist, int64_t *p_stride,
                                int64_t *p_kernel_size, int64_t *p_dilation,
                                cudaStream_t stream, long D, void **metadata);
