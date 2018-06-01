// All CPP functions that will be used in sparse.c
#include <cuda_runtime.h>

extern int _initialize_coords(int *coords, int nrows, int *p_pixel_dist, int D,
                              void **metadata);

extern int _initialize_coords_with_duplicates(int *coords, int nrows,
                                              int *p_pixel_dist, int D,
                                              void **metadata);

extern int _initialize_out_coords(int *p_pixel_dist, int *p_stride,
                                  bool is_transpose, int D, void **metadata);

extern int _initialize_origin_coords(int *p_pixel_dist, int batch_size, int D,
                                     void **metadata);

extern int _get_index_map(int *coords, int nrows, int *p_index_map,
                          int index_map_nrows, int *p_pixel_dist, int D,
                          void **metadata);

extern int _get_coords(int *coords, int *p_pixel_dist, int D, void **metadata);

extern int _get_num_coords(int *p_pixel_dist, int *p_nrows, int D,
                           void **metadata);

extern int _get_permutation(int *p_permutation, int *p_pixel_dist_src,
                            int *p_pixel_dist_dst, int D, void **metadata);

extern void _clear(int D, void **metadata);

extern int _conv_fw(float *p_in_feat, int in_nchannel, float *p_out_feat,
                    int out_nchannel, float *p_kernel, int out_nrows,
                    int *p_pixel_dist, int *p_stride, int *p_kernel_size,
                    int *p_dilation, int region_type, int *p_offset,
                    int n_offset, int D, void **metadata);

extern int _conv_tr_fw(float *p_in_feat, int in_nchannel, float *p_out_feat,
                       int out_nchannel, float *p_kernel, int out_nrows,
                       int *p_pixel_dist, int *p_stride, int *p_kernel_size,
                       int *p_dilation, int region_type, int *p_offset,
                       int n_offset, int D, void **metadata);

extern int _conv_bw(float *p_in_feat, float *p_grad_in_feat, int in_nchannel,
                    float *p_grad_out_feat, int out_nchannel, float *p_kernel,
                    float *p_grad_kernel, int out_nrows, int *p_pixel_dist,
                    int *p_stride, int *p_kernel_size, int *p_dilation, int D,
                    void **metadata);

extern int _conv_tr_bw(float *p_in_feat, float *p_grad_in_feat, int in_nchannel,
                       float *p_grad_out_feat, int out_nchannel,
                       float *p_kernel, float *p_grad_kernel, int out_nrows,
                       int *p_pixel_dist, int *p_stride, int *p_kernel_size,
                       int *p_dilation, int D, void **metadata);

extern int _conv_fw_gpu(float *d_in_feat, int in_nchannel, float *d_out_feat,
                        int out_nchannel, float *d_kernel, int out_nrows,
                        int *p_pixel_dist, int *p_stride, int *p_kernel_size,
                        int *p_dilation, int region_type, int *p_offset,
                        int n_offset, cudaStream_t stream, int D,
                        void **metadata);

extern int _conv_tr_fw_gpu(float *d_in_feat, int in_nchannel, float *d_out_feat,
                           int out_nchannel, float *d_kernel, int out_nrows,
                           int *p_pixel_dist, int *p_stride, int *p_kernel_size,
                           int *p_dilation, int region_type, int *p_offset,
                           int n_offset, cudaStream_t stream, int D,
                           void **metadata);

extern int _conv_bw_gpu(float *d_in_feat, float *d_grad_in_feat,
                        int in_nchannel, float *d_grad_out_feat,
                        int out_nchannel, float *d_kernel, float *d_grad_kernel,
                        int out_nrows, int *p_pixel_dist, int *p_stride,
                        int *p_kernel_size, int *p_dilation,
                        cudaStream_t stream, int D, void **metadata);

extern int _conv_tr_bw_gpu(float *d_in_feat, float *d_grad_in_feat,
                           int in_nchannel, float *d_grad_out_feat,
                           int out_nchannel, float *d_kernel,
                           float *d_grad_kernel, int out_nrows,
                           int *p_pixel_dist, int *p_stride, int *p_kernel_size,
                           int *p_dilation, cudaStream_t stream, int D,
                           void **metadata);

extern int _max_pooling_fw(float *p_in_feat, float *p_out_feat,
                           int *p_mask_index, int nchannel,
                           int out_nrows, int *p_pixel_dist,
                           int *p_stride, int *p_kernel_size,
                           int *p_dilation, int region_type,
                           int *p_offset, int n_offset, int D,
                           void **metadata);

extern int _max_pooling_bw(float *p_grad_in_feat, int in_nrows,
                           float *p_grad_out_feat, int out_nrows,
                           int *p_mask_index, int nchannel,
                           int *p_pixel_dist, int *p_stride,
                           int *p_kernel_size, int *p_dilation, int D,
                           void **metadata);

extern int _max_pooling_fw_gpu(float *d_in_feat, float *d_out_feat,
                               int out_nrows, int *d_mask_index,
                               int nchannel, int *p_pixel_dist,
                               int *p_stride, int *p_kernel_size,
                               int *p_dilation, int region_type,
                               int *p_offset, int n_offset,
                               cudaStream_t stream, int D, void **metadata);

extern int _max_pooling_bw_gpu(float *d_grad_in_feat, int in_nrows,
                               float *d_grad_out_feat, int out_nrows,
                               int *d_mask_index, int nchannel,
                               int *p_pixel_dist, int *p_stride,
                               int *p_kernel_size, int *p_dilation,
                               cudaStream_t stream, int D, void **metadata);

extern int _nonzero_avg_pooling_fw(float *p_in_feat, float *p_out_feat,
                                   int *p_num_nonzero, int nchannel,
                                   int out_nrows, int *p_pixel_dist,
                                   int *p_stride, int *p_kernel_size,
                                   int *p_dilation, int region_type,
                                   int *p_offset, int n_offset, int D,
                                   void **metadata);

extern int _nonzero_avg_pooling_bw(float *p_grad_in_feat, int in_nrows,
                                   float *p_grad_out_feat, int out_nrows,
                                   int *p_num_nonzero, int nchannel,
                                   int *p_pixel_dist, int *p_stride,
                                   int *p_kernel_size, int *p_dilation, int D,
                                   void **metadata);

extern int
_nonzero_avg_pooling_fw_gpu(float *d_in_feat, float *d_out_feat, int out_nrows,
                            int *d_num_nonzero, int nchannel, int *p_pixel_dist,
                            int *p_stride, int *p_kernel_size, int *p_dilation,
                            int region_type, int *p_offset, int n_offset,
                            cudaStream_t stream, int D, void **metadata);

extern int _nonzero_avg_pooling_bw_gpu(float *d_grad_in_feat, int in_nrows,
                                       float *d_grad_out_feat, int out_nrows,
                                       int *d_num_nonzero, int nchannel,
                                       int *p_pixel_dist, int *p_stride,
                                       int *p_kernel_size, int *p_dilation,
                                       cudaStream_t stream, int D,
                                       void **metadata);

extern int _global_avg_pooling_fw(float *p_in_feat, float *p_out_feat,
                                  int out_nrows, int nchannel,
                                  int *p_num_nonzero, int *p_pixel_dist, int D,
                                  void **metadata);

extern int _global_avg_pooling_bw(float *p_grad_in_feat, int in_nrows,
                                  float *p_grad_out_feat, int out_nrows,
                                  int nchannel, int *p_num_nonzero,
                                  int *p_pixel_dist, int D, void **metadata);

extern int _global_avg_pooling_fw_gpu(float *d_in_feat, float *d_out_feat,
                                      int out_nrows, int nchannel,
                                      int *d_num_nonzero, int *p_pixel_dist,
                                      cudaStream_t stream, int D,
                                      void **metadata);

extern int _global_avg_pooling_bw_gpu(float *d_grad_in_feat, int in_nrows,
                                      float *d_grad_out_feat, int out_nrows,
                                      int nchannel, int *d_num_nonzero,
                                      int *p_pixel_dist, cudaStream_t stream,
                                      int D, void **metadata);
