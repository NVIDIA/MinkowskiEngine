#ifndef EXTERNS
#define EXTERNS
#include <cstdio>

#include "src/gpu.cuh"
#include "src/main.hpp"

#define NOARG

#define SWITCH_DIM(RETURN, func, ...)                                          \
  switch (D) {                                                                 \
  case 2:                                                                      \
    RETURN func<2>(__VA_ARGS__);                                               \
    break;                                                                     \
  case 3:                                                                      \
    RETURN func<3>(__VA_ARGS__);                                               \
    break;                                                                     \
  case 4:                                                                      \
    RETURN func<4>(__VA_ARGS__);                                               \
    break;                                                                     \
  case 5:                                                                      \
    RETURN func<5>(__VA_ARGS__);                                               \
    break;                                                                     \
  case 6:                                                                      \
    RETURN func<6>(__VA_ARGS__);                                               \
    break;                                                                     \
  case 7:                                                                      \
    RETURN func<7>(__VA_ARGS__);                                               \
    break;                                                                     \
  case 8:                                                                      \
    RETURN func<8>(__VA_ARGS__);                                               \
    break;                                                                     \
  case 9:                                                                      \
    RETURN func<9>(__VA_ARGS__);                                               \
    break;                                                                     \
  case 10:                                                                     \
    RETURN func<10>(__VA_ARGS__);                                              \
    break;                                                                     \
  case 11:                                                                     \
    RETURN func<11>(__VA_ARGS__);                                              \
    break;                                                                     \
  case 12:                                                                     \
    RETURN func<12>(__VA_ARGS__);                                              \
    break;                                                                     \
  default:                                                                     \
    printf("%s\n", "Dimension mismatch");                                      \
  }

template <uint8_t D>
long t_initialize_coords(const int64_t *coords, int64_t nrows,
                         const int64_t *p_pixel_dist, void **metadata);
extern "C" long _initialize_coords(long *coords, long nrows, long *p_pixel_dist,
                                   long D, void **metadata) {
  SWITCH_DIM(return, t_initialize_coords, coords, nrows, p_pixel_dist, metadata)
}

template <uint8_t D>
long t_initialize_out_coords(const int64_t *p_pixel_dist,
                             const int64_t *p_stride, bool is_transpose,
                             void **metadata);
extern "C" long _initialize_out_coords(long *p_pixel_dist, long *p_stride,
                                       bool is_transpose, long D,
                                       void **metadata) {
  SWITCH_DIM(return, t_initialize_out_coords, p_pixel_dist, p_stride,
                   is_transpose, metadata)
}

template <uint8_t D>
long t_initialize_coords_with_duplicates(const int64_t *coords, int64_t nrows,
                                         const int64_t *p_pixel_dist,
                                         void **metadata);
extern "C" long _initialize_coords_with_duplicates(long *coords, long nrows,
                                                   long *p_pixel_dist, long D,
                                                   void **metadata) {
  SWITCH_DIM(return, t_initialize_coords_with_duplicates, coords, nrows,
                   p_pixel_dist, metadata)
}

template <uint8_t D>
long t_get_index_map(const int64_t *coords, int64_t nrows, int64_t *p_index_map,
                     int64_t index_map_nrows, const int64_t *p_pixel_dist,
                     void **metadata);
extern "C" long _get_index_map(long *coords, long nrows, long *p_index_map,
                               long index_map_nrows, long *p_pixel_dist, long D,
                               void **metadata) {
  SWITCH_DIM(return, t_get_index_map, coords, nrows, p_index_map,
                   index_map_nrows, p_pixel_dist, metadata)
}

template <uint8_t D>
long t_get_num_coords(const long *p_pixel_dist, int64_t *nrows,
                      void **metadata);
extern "C" long _get_num_coords(long *p_pixel_dist, long *p_nrows, long D,
                                void **metadata) {
  SWITCH_DIM(return, t_get_num_coords, p_pixel_dist, p_nrows, metadata)
}

template <uint8_t D>
long t_get_coords(long *coords, const int64_t *p_pixel_dist, void **metadata);
extern "C" long _get_coords(long *coords, long *p_pixel_dist, long D,
                            void **metadata) {
  SWITCH_DIM(return, t_get_coords, coords, p_pixel_dist, metadata)
}

template <uint8_t D>
long t_get_permutation(long *p_permutation, const int64_t *p_pixel_dist_src,
                       const int64_t *p_pixel_dist_dst, void **metadata);
extern "C" long _get_permutation(long *p_permutation, long *p_pixel_dist_src,
                                 long *p_pixel_dist_dst, long D,
                                 void **metadata) {
  SWITCH_DIM(return, t_get_permutation, p_permutation, p_pixel_dist_src,
                   p_pixel_dist_dst, metadata)
}

template <uint8_t D> void t_clear(void **metadata);
extern "C" void _clear(long D, void **metadata) {
  SWITCH_DIM(NOARG, t_clear, metadata)
}

template <uint8_t D>
long t_conv_fw(const float *p_in_feat, int64_t in_nchannel, float *p_out_feat,
               int64_t out_nchannel, const float *p_kernel, int64_t out_nrows,
               const int64_t *p_pixel_dist, const int64_t *p_stride,
               const int64_t *p_kernel_size, const int64_t *p_dilation,
               int64_t region_type, const int64_t *p_offset, int64_t n_offset,
               void **metadata);
extern "C" long _conv_fw(float *p_in_feat, long in_nchannel, float *p_out_feat,
                         long out_nchannel, float *p_kernel, long out_nrows,
                         long *p_pixel_dist, long *p_stride,
                         long *p_kernel_size, long *p_dilation,
                         long region_type, long *p_offset, long n_offset,
                         long D, void **metadata) {
  SWITCH_DIM(return, t_conv_fw, p_in_feat, in_nchannel, p_out_feat,
                   out_nchannel, p_kernel, out_nrows, p_pixel_dist, p_stride,
                   p_kernel_size, p_dilation, region_type, p_offset, n_offset,
                   metadata)
}

template <uint8_t D>
long t_conv_tr_fw(const float *p_in_feat, int64_t in_nchannel,
                  float *p_out_feat, int64_t out_nchannel,
                  const float *p_kernel, int64_t out_nrows,
                  const int64_t *p_pixel_dist, const int64_t *p_stride,
                  const int64_t *p_kernel_size, const int64_t *p_dilation,
                  int64_t region_type, const int64_t *p_offset,
                  int64_t n_offset, void **metadata);
extern "C" long _conv_tr_fw(float *p_in_feat, long in_nchannel,
                            float *p_out_feat, long out_nchannel,
                            float *p_kernel, long out_nrows, long *p_pixel_dist,
                            long *p_stride, long *p_kernel_size,
                            long *p_dilation, long region_type, long *p_offset,
                            long n_offset, long D, void **metadata) {
  SWITCH_DIM(return, t_conv_tr_fw, p_in_feat, in_nchannel, p_out_feat,
                   out_nchannel, p_kernel, out_nrows, p_pixel_dist, p_stride,
                   p_kernel_size, p_dilation, region_type, p_offset, n_offset,
                   metadata)
}

template <uint8_t D>
long t_conv_bw(const float *p_in_feat, float *p_grad_in_feat,
               int64_t in_nchannel, const float *p_grad_out_feat,
               int64_t out_nchannel, const float *p_kernel,
               float *p_grad_kernel, int64_t out_nrows,
               const int64_t *p_pixel_dist, const int64_t *p_stride,
               const int64_t *p_kernel_size, const int64_t *p_dilation,
               void **metadata);
extern "C" long _conv_bw(float *p_in_feat, float *p_grad_in_feat,
                         long in_nchannel, float *p_grad_out_feat,
                         long out_nchannel, float *p_kernel,
                         float *p_grad_kernel, long out_nrows,
                         long *p_pixel_dist, long *p_stride,
                         long *p_kernel_size, long *p_dilation, long D,
                         void **metadata) {
  SWITCH_DIM(return, t_conv_bw, p_in_feat, p_grad_in_feat, in_nchannel,
                   p_grad_out_feat, out_nchannel, p_kernel, p_grad_kernel,
                   out_nrows, p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                   metadata)
}

template <uint8_t D>
long t_conv_tr_bw(const float *p_in_feat, float *p_grad_in_feat,
                  int64_t in_nchannel, const float *p_grad_out_feat,
                  int64_t out_nchannel, const float *p_kernel,
                  float *p_grad_kernel, int64_t out_nrows,
                  const int64_t *p_pixel_dist, const int64_t *p_stride,
                  const int64_t *p_kernel_size, const int64_t *p_dilation,
                  void **metadata);
extern "C" long _conv_tr_bw(float *p_in_feat, float *p_grad_in_feat,
                            long in_nchannel, float *p_grad_out_feat,
                            long out_nchannel, float *p_kernel,
                            float *p_grad_kernel, long out_nrows,
                            long *p_pixel_dist, long *p_stride,
                            long *p_kernel_size, long *p_dilation, long D,
                            void **metadata) {
  SWITCH_DIM(return, t_conv_tr_bw, p_in_feat, p_grad_in_feat, in_nchannel,
                   p_grad_out_feat, out_nchannel, p_kernel, p_grad_kernel,
                   out_nrows, p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                   metadata)
}

template <uint8_t D>
long t_conv_fw_gpu(const float *d_in_feat, int64_t in_nchannel,
                   float *d_out_feat, int64_t out_nchannel,
                   const float *d_kernel, int64_t out_nrows,
                   const int64_t *p_pixel_dist, const int64_t *p_stride,
                   const int64_t *p_kernel_size, const int64_t *p_dilation,
                   int64_t region_type, const int64_t *p_offset,
                   int64_t n_offset, cudaStream_t stream, void **metadata);
extern "C" long _conv_fw_gpu(float *d_in_feat, long in_nchannel,
                             float *d_out_feat, long out_nchannel,
                             float *d_kernel, long out_nrows,
                             long *p_pixel_dist, long *p_stride,
                             long *p_kernel_size, long *p_dilation,
                             long region_type, long *p_offset, long n_offset,
                             cudaStream_t stream, long D, void **metadata) {
  SWITCH_DIM(return, t_conv_fw_gpu, d_in_feat, in_nchannel, d_out_feat,
                   out_nchannel, d_kernel, out_nrows, p_pixel_dist, p_stride,
                   p_kernel_size, p_dilation, region_type, p_offset, n_offset,
                   stream, metadata)
}

template <uint8_t D>
long t_conv_tr_fw_gpu(const float *d_in_feat, int64_t in_nchannel,
                      float *d_out_feat, int64_t out_nchannel,
                      const float *d_kernel, int64_t out_nrows,
                      const int64_t *p_pixel_dist, const int64_t *p_stride,
                      const int64_t *p_kernel_size, const int64_t *p_dilation,
                      int64_t region_type, const int64_t *p_offset,
                      int64_t n_offset, cudaStream_t stream, void **metadata);
extern "C" long _conv_tr_fw_gpu(float *d_in_feat, long in_nchannel,
                                float *d_out_feat, long out_nchannel,
                                float *d_kernel, long out_nrows,
                                long *p_pixel_dist, long *p_stride,
                                long *p_kernel_size, long *p_dilation,
                                long region_type, long *p_offset, long n_offset,
                                cudaStream_t stream, long D, void **metadata) {
  SWITCH_DIM(return, t_conv_tr_fw_gpu, d_in_feat, in_nchannel, d_out_feat,
                   out_nchannel, d_kernel, out_nrows, p_pixel_dist, p_stride,
                   p_kernel_size, p_dilation, region_type, p_offset, n_offset,
                   stream, metadata)
}

template <uint8_t D>
long t_conv_bw_gpu(const float *d_in_feat, float *d_grad_in_feat,
                   int64_t in_nchannel, const float *d_grad_out_feat,
                   int64_t out_nchannel, const float *d_kernel,
                   float *d_grad_kernel, int64_t out_nrows,
                   const int64_t *p_pixel_dist, const int64_t *p_stride,
                   const int64_t *p_kernel_size, const int64_t *p_dilation,
                   cudaStream_t stream, void **metadata);
extern "C" long _conv_bw_gpu(float *d_in_feat, float *d_grad_in_feat,
                             long in_nchannel, float *d_grad_out_feat,
                             long out_nchannel, float *d_kernel,
                             float *d_grad_kernel, long out_nrows,
                             long *p_pixel_dist, long *p_stride,
                             long *p_kernel_size, long *p_dilation,
                             cudaStream_t stream, long D, void **metadata) {
  SWITCH_DIM(return, t_conv_bw_gpu, d_in_feat, d_grad_in_feat, in_nchannel,
                   d_grad_out_feat, out_nchannel, d_kernel, d_grad_kernel,
                   out_nrows, p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                   stream, metadata);
}

template <uint8_t D>
long t_conv_tr_bw_gpu(const float *d_in_feat, float *d_grad_in_feat,
                      int64_t in_nchannel, const float *d_grad_out_feat,
                      int64_t out_nchannel, const float *d_kernel,
                      float *d_grad_kernel, int64_t out_nrows,
                      const int64_t *p_pixel_dist, const int64_t *p_stride,
                      const int64_t *p_kernel_size, const int64_t *p_dilation,
                      cudaStream_t stream, void **metadata);
extern "C" long _conv_tr_bw_gpu(float *d_in_feat, float *d_grad_in_feat,
                                long in_nchannel, float *d_grad_out_feat,
                                long out_nchannel, float *d_kernel,
                                float *d_grad_kernel, long out_nrows,
                                long *p_pixel_dist, long *p_stride,
                                long *p_kernel_size, long *p_dilation,
                                cudaStream_t stream, long D, void **metadata) {
  SWITCH_DIM(return, t_conv_tr_bw_gpu, d_in_feat, d_grad_in_feat, in_nchannel,
                   d_grad_out_feat, out_nchannel, d_kernel, d_grad_kernel,
                   out_nrows, p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                   stream, metadata);
}

template <uint8_t D>
long t_max_pooling_fw(const float *p_in_feat, float *p_out_feat,
                      int64_t *p_mask_index, int64_t nchannel,
                      int64_t out_nrows, const int64_t *p_pixel_dist,
                      const int64_t *p_stride, const int64_t *p_kernel_size,
                      const int64_t *p_dilation, int64_t region_type,
                      const int64_t *p_offset, int64_t n_offset,
                      void **metadata);
extern "C" long _max_pooling_fw(float *p_in_feat, float *p_out_feat,
                                long *p_mask_index, long nchannel,
                                long out_nrows, long *p_pixel_dist,
                                long *p_stride, long *p_kernel_size,
                                long *p_dilation, long region_type,
                                long *p_offset, long n_offset, long D,
                                void **metadata) {
  SWITCH_DIM(return, t_max_pooling_fw, p_in_feat, p_out_feat, p_mask_index,
                   nchannel, out_nrows, p_pixel_dist, p_stride, p_kernel_size,
                   p_dilation, region_type, p_offset, n_offset, metadata);
}

template <uint8_t D>
long t_max_pooling_bw(float *p_grad_in_feat, int64_t in_nrows,
                      float *p_grad_out_feat, int64_t out_nrows,
                      const int64_t *p_mask_index, int64_t nchannel,
                      const int64_t *p_pixel_dist, const int64_t *p_stride,
                      const int64_t *p_kernel_size, const int64_t *p_dilation,
                      void **metadata);
extern "C" long _max_pooling_bw(float *p_grad_in_feat, long in_nrows,
                                float *p_grad_out_feat, long out_nrows,
                                long *p_mask_index, long nchannel,
                                long *p_pixel_dist, long *p_stride,
                                long *p_kernel_size, long *p_dilation, long D,
                                void **metadata) {
  SWITCH_DIM(return, t_max_pooling_bw, p_grad_in_feat, in_nrows,
                   p_grad_out_feat, out_nrows, p_mask_index, nchannel,
                   p_pixel_dist, p_stride, p_kernel_size, p_dilation, metadata);
}

template <uint8_t D>
long t_max_pooling_fw_gpu(const float *d_in_feat, float *d_out_feat,
                          int64_t out_nrows, int64_t *d_mask_index,
                          int64_t nchannel, const int64_t *p_pixel_dist,
                          const int64_t *p_stride, const int64_t *p_kernel_size,
                          const int64_t *p_dilation, int64_t region_type,
                          const int64_t *p_offset, int64_t n_offset,
                          cudaStream_t stream, void **metadata);
extern "C" long
_max_pooling_fw_gpu(float *d_in_feat, float *d_out_feat, long out_nrows,
                    long *d_mask_index, long nchannel, long *p_pixel_dist,
                    long *p_stride, long *p_kernel_size, long *p_dilation,
                    long region_type, long *p_offset, long n_offset,
                    cudaStream_t stream, long D, void **metadata) {
  SWITCH_DIM(return, t_max_pooling_fw_gpu, d_in_feat, d_out_feat, out_nrows,
                   d_mask_index, nchannel, p_pixel_dist, p_stride,
                   p_kernel_size, p_dilation, region_type, p_offset, n_offset,
                   stream, metadata);
}

template <uint8_t D>
long t_max_pooling_bw_gpu(float *d_grad_in_feat, int64_t in_nrows,
                          const float *d_grad_out_feat, int64_t out_nrows,
                          const int64_t *d_mask_index, int64_t nchannel,
                          const int64_t *p_pixel_dist, const int64_t *p_stride,
                          const int64_t *p_kernel_size,
                          const int64_t *p_dilation, cudaStream_t stream,
                          void **metadata);
extern "C" long _max_pooling_bw_gpu(float *d_grad_in_feat, long in_nrows,
                                    float *d_grad_out_feat, long out_nrows,
                                    long *d_mask_index, long nchannel,
                                    long *p_pixel_dist, long *p_stride,
                                    long *p_kernel_size, long *p_dilation,
                                    cudaStream_t stream, long D,
                                    void **metadata) {
  SWITCH_DIM(return, t_max_pooling_bw_gpu, d_grad_in_feat, in_nrows,
                   d_grad_out_feat, out_nrows, d_mask_index, nchannel,
                   p_pixel_dist, p_stride, p_kernel_size, p_dilation, stream,
                   metadata);
}
#endif
