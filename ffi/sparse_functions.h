// All CPP functions that will be used in sparse.c
#include <cuda_runtime.h>

extern long _initialize_coords(int *coords, int nrows, int *p_pixel_dist, int D,
                               void **metadata);

extern long _initialize_coords_with_duplicates(int *coords, int nrows,
                                               int *p_pixel_dist, int D,
                                               void **metadata);

extern long _initialize_out_coords(uint64_t *p_in_coords_key,
                                   uint64_t *p_out_coords_key,
                                   int *p_pixel_dist, int *p_stride,
                                   bool is_transpose, int D, void **metadata);

extern long _initialize_valid_conv_out_coords(
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, int *p_pixel_dist,
    int *p_stride, int *p_kernel_size, int *p_dilation, bool is_transpose,
    int D, void **metadata);

extern long _initialize_origin_coords(uint64_t *p_in_coords_key,
                                      int *p_pixel_dist, int batch_size, int D,
                                      void **metadata);

extern long _get_index_map(int *coords, int nrows, int *p_index_map,
                           int index_map_nrows, int *p_pixel_dist, int D,
                           void **metadata);

extern long _get_coords(int *coords, uint64_t *p_coords_key, int *p_pixel_dist,
                        int D, void **metadata);

extern long _get_num_coords(uint64_t *p_coords_key, int *p_pixel_dist,
                            int *p_nrows, int D, void **metadata);

extern long _get_permutation(int *p_permutation, int *p_pixel_dist_src,
                             int *p_pixel_dist_dst, int D, void **metadata);

extern void _clear(int D, void **metadata);

extern long _conv_fw(float *p_in_feat, int in_nchannel, float *p_out_feat,
                     int out_nchannel, float *p_kernel, int out_nrows,
                     int *p_pixel_dist, int *p_stride, int *p_kernel_size,
                     int *p_dilation, int region_type, int *p_offset,
                     int n_offset, uint64_t *p_in_coords_key,
                     uint64_t *p_out_coords_key, int D, void **metadata);

extern long _conv_bw(float *p_in_feat, float *p_grad_in_feat, int in_nchannel,
                     float *p_grad_out_feat, int out_nchannel, float *p_kernel,
                     float *p_grad_kernel, int out_nrows, int *p_pixel_dist,
                     int *p_stride, int *p_kernel_size, int *p_dilation,
                     uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                     int D, void **metadata);

extern long _conv_tr_fw(float *p_in_feat, int in_nchannel, float *p_out_feat,
                        int out_nchannel, float *p_kernel, int out_nrows,
                        int *p_pixel_dist, int *p_stride, int *p_kernel_size,
                        int *p_dilation, int region_type, int *p_offset,
                        int n_offset, uint64_t *p_in_coords_key,
                        uint64_t *p_out_coords_key, int D, void **metadata);

extern long _conv_tr_bw(float *p_in_feat, float *p_grad_in_feat,
                        int in_nchannel, float *p_grad_out_feat,
                        int out_nchannel, float *p_kernel, float *p_grad_kernel,
                        int out_nrows, int *p_pixel_dist, int *p_stride,
                        int *p_kernel_size, int *p_dilation,
                        uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                        int D, void **metadata);

extern long _conv_fw_gpu(float *d_in_feat, int in_nchannel, float *d_out_feat,
                         int out_nchannel, float *d_kernel, int out_nrows,
                         int *p_pixel_dist, int *p_stride, int *p_kernel_size,
                         int *p_dilation, int region_type, int *p_offset,
                         int n_offset, uint64_t *p_in_coords_key,
                         uint64_t *p_out_coords_key, cudaStream_t stream, int D,
                         void **metadata);

extern long _conv_tr_fw_gpu(float *d_in_feat, int in_nchannel,
                            float *d_out_feat, int out_nchannel,
                            float *d_kernel, int out_nrows, int *p_pixel_dist,
                            int *p_stride, int *p_kernel_size, int *p_dilation,
                            int region_type, int *p_offset, int n_offset,
                            uint64_t *p_in_coords_key,
                            uint64_t *p_out_coords_key, cudaStream_t stream,
                            int D, void **metadata);

extern long _conv_bw_gpu(float *d_in_feat, float *d_grad_in_feat,
                         int in_nchannel, float *d_grad_out_feat,
                         int out_nchannel, float *d_kernel,
                         float *d_grad_kernel, int out_nrows, int *p_pixel_dist,
                         int *p_stride, int *p_kernel_size, int *p_dilation,
                         uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                         cudaStream_t stream, int D, void **metadata);

extern long
_conv_tr_bw_gpu(float *d_in_feat, float *d_grad_in_feat, int in_nchannel,
                float *d_grad_out_feat, int out_nchannel, float *d_kernel,
                float *d_grad_kernel, int out_nrows, int *p_pixel_dist,
                int *p_stride, int *p_kernel_size, int *p_dilation,
                uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                cudaStream_t stream, int D, void **metadata);

extern long _max_pooling_fw(float *p_in_feat, float *p_out_feat,
                            int *p_mask_index, int nchannel, int out_nrows,
                            int *p_pixel_dist, int *p_stride,
                            int *p_kernel_size, int *p_dilation,
                            int region_type, int *p_offset, int n_offset,
                            uint64_t *p_in_coords_key,
                            uint64_t *p_out_coords_key, int D, void **metadata);

extern long _max_pooling_bw(float *p_grad_in_feat, int in_nrows,
                            float *p_grad_out_feat, int out_nrows,
                            int *p_mask_index, int nchannel, int *p_pixel_dist,
                            int *p_stride, int *p_kernel_size, int *p_dilation,
                            uint64_t *p_in_coords_key,
                            uint64_t *p_out_coords_key, int D, void **metadata);

extern long _max_pooling_fw_gpu(float *d_in_feat, float *d_out_feat,
                                int out_nrows, int *d_mask_index, int nchannel,
                                int *p_pixel_dist, int *p_stride,
                                int *p_kernel_size, int *p_dilation,
                                int region_type, int *p_offset, int n_offset,
                                uint64_t *p_in_coords_key,
                                uint64_t *p_out_coords_key, cudaStream_t stream,
                                int D, void **metadata);

extern long _max_pooling_bw_gpu(
    float *d_grad_in_feat, int in_nrows, float *d_grad_out_feat, int out_nrows,
    int *d_mask_index, int nchannel, int *p_pixel_dist, int *p_stride,
    int *p_kernel_size, int *p_dilation, uint64_t *p_in_coords_key,
    uint64_t *p_out_coords_key, cudaStream_t stream, int D, void **metadata);

extern long _nonzero_avg_pooling_fw(
    float *p_in_feat, float *p_out_feat, float *p_num_nonzero, int nchannel,
    int out_nrows, int *p_pixel_dist, int *p_stride, int *p_kernel_size,
    int *p_dilation, int region_type, int *p_offset, int n_offset,
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, int use_avg, int D,
    void **metadata);

extern long _nonzero_avg_pooling_bw(
    float *p_grad_in_feat, int in_nrows, float *p_grad_out_feat, int out_nrows,
    float *p_num_nonzero, int nchannel, int *p_pixel_dist, int *p_stride,
    int *p_kernel_size, int *p_dilation, uint64_t *p_in_coords_key,
    uint64_t *p_out_coords_key, int use_avg, int D, void **metadata);

extern long _nonzero_avg_pooling_fw_gpu(
    float *d_in_feat, int in_nrows, float *d_out_feat, int out_nrows,
    float *d_num_nonzero, int nchannel, int *p_pixel_dist, int *p_stride,
    int *p_kernel_size, int *p_dilation, int region_type, int *p_offset,
    int n_offset, uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
    int use_avg, cudaStream_t stream, int D, void **metadata);

extern long _nonzero_avg_pooling_bw_gpu(
    float *d_grad_in_feat, int in_nrows, float *d_grad_out_feat, int out_nrows,
    float *d_num_nonzero, int nchannel, int *p_pixel_dist, int *p_stride,
    int *p_kernel_size, int *p_dilation, uint64_t *p_in_coords_key,
    uint64_t *p_out_coords_key, int use_avg, cudaStream_t stream, int D,
    void **metadata);

extern long _unpooling_fw(float *p_in_feat, float *p_out_feat,
                          float *p_num_nonzero, int nchannel, int out_nrows,
                          int *p_pixel_dist, int *p_stride, int *p_kernel_size,
                          int *p_dilation, int region_type, int *p_offset,
                          int n_offset, uint64_t *p_in_coords_key,
                          uint64_t *p_out_coords_key, int D, void **metadata);

extern long _unpooling_bw(float *p_grad_in_feat, int in_nrows,
                          float *p_grad_out_feat, int out_nrows,
                          float *p_num_nonzero, int nchannel, int *p_pixel_dist,
                          int *p_stride, int *p_kernel_size, int *p_dilation,
                          uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                          int D, void **metadata);

extern long _unpooling_fw_gpu(float *d_in_feat, int in_nrows, float *d_out_feat,
                              int out_nrows, float *d_num_nonzero, int nchannel,
                              int *p_pixel_dist, int *p_stride,
                              int *p_kernel_size, int *p_dilation,
                              int region_type, int *p_offset, int n_offset,
                              uint64_t *p_in_coords_key,
                              uint64_t *p_out_coords_key, cudaStream_t stream,
                              int D, void **metadata);

extern long _unpooling_bw_gpu(
    float *d_grad_in_feat, int in_nrows, float *d_grad_out_feat, int out_nrows,
    float *d_num_nonzero, int nchannel, int *p_pixel_dist, int *p_stride,
    int *p_kernel_size, int *p_dilation, uint64_t *p_in_coords_key,
    uint64_t *p_out_coords_key, cudaStream_t stream, int D, void **metadata);

extern long _global_avg_pooling_fw(float *p_in_feat, float *p_out_feat,
                                   int out_nrows, int nchannel,
                                   float *p_num_nonzero, int *p_pixel_dist,
                                   uint64_t *p_in_coords_key,
                                   uint64_t *p_out_coords_key, int D,
                                   void **metadata);

extern long _global_avg_pooling_bw(float *p_grad_in_feat, int in_nrows,
                                   float *p_grad_out_feat, int out_nrows,
                                   int nchannel, float *p_num_nonzero,
                                   int *p_pixel_dist, uint64_t *p_in_coords_key,
                                   uint64_t *p_out_coords_key, int D,
                                   void **metadata);

extern long
_global_avg_pooling_fw_gpu(float *d_in_feat, int in_nrows, float *d_out_feat,
                           int out_nrows, int nchannel, float *d_num_nonzero,
                           int *p_pixel_dist, uint64_t *p_in_coords_key,
                           uint64_t *p_out_coords_key, cudaStream_t stream,
                           int D, void **metadata);

extern long _global_avg_pooling_bw_gpu(
    float *d_grad_in_feat, int in_nrows, float *d_grad_out_feat, int out_nrows,
    int nchannel, float *d_num_nonzero, int *p_pixel_dist,
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, cudaStream_t stream,
    int D, void **metadata);

extern long
_global_broadcast_fw(float *p_in_feat, int in_nrows, float *p_in_feat_global,
                     int in_nrows_global, float *p_out_feat, int nchannel,
                     int *p_pixel_dist, int op, uint64_t *p_in_coords_key,
                     uint64_t *p_out_coords_key, int D, void **metadata);

extern long
_global_broadcast_bw(float *p_in_feat, float *p_grad_in_feat, int in_nrows,
                     float *p_in_feat_global, float *p_grad_in_feat_global,
                     int in_nrows_global, float *p_grad_out_feat, int nchannel,
                     int *p_pixel_dist, int op, uint64_t *p_in_coords_key,
                     uint64_t *p_out_coords_key, int D, void **metadata);

extern long _global_broadcast_fw_gpu(
    float *p_in_feat, int in_nrows, float *p_in_feat_global,
    int in_nrows_global, float *p_out_feat, int nchannel, int *p_pixel_dist,
    int op, uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
    cudaStream_t stream, int D, void **metadata);

extern long
_global_broadcast_bw_gpu(float *p_in_feat, float *p_grad_in_feat, int in_nrows,
                         float *p_in_feat_global, float *p_grad_in_feat_global,
                         int in_nrows_global, float *p_grad_out_feat,
                         int nchannel, int *p_pixel_dist, int op,
                         uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                         cudaStream_t stream, int D, void **metadata);
