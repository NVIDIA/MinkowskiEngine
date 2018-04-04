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
                         int64_t pixel_dist, void **metadata);
extern "C" long _initialize_coords(long *coords, long nrows, long pixel_dist,
                                   long D, void **metadata) {
  SWITCH_DIM(return, t_initialize_coords, coords, nrows, pixel_dist, metadata)
}

template <uint8_t D>
long t_initialize_out_coords(int64_t pixel_dist, int64_t stride,
                             bool is_transpose, void **metadata);
extern "C" long _initialize_out_coords(long pixel_dist, long stride,
                                       bool is_transpose, long D,
                                       void **metadata) {
  SWITCH_DIM(return, t_initialize_out_coords, pixel_dist, stride, is_transpose,
                   metadata)
}

template <uint8_t D>
long t_initialize_coords_with_duplicates(const int64_t *coords, int64_t nrows,
                                         int64_t pixel_dist, void **metadata);
extern "C" long _initialize_coords_with_duplicates(long *coords, long nrows,
                                                   long pixel_dist, long D,
                                                   void **metadata) {
  SWITCH_DIM(return, t_initialize_coords_with_duplicates, coords, nrows,
                   pixel_dist, metadata)
}

template <uint8_t D>
long t_get_index_map(const int64_t *coords, int64_t nrows, int64_t *p_index_map,
                     int64_t index_map_nrows, int64_t pixel_dist,
                     void **metadata);
extern "C" long _get_index_map(const long *coords, long nrows,
                               long *p_index_map, long index_map_nrows,
                               long pixel_dist, long D, void **metadata) {
  SWITCH_DIM(return, t_get_index_map, coords, nrows, p_index_map,
                   index_map_nrows, pixel_dist, metadata)
}

template <uint8_t D>
long t_get_num_coords(long pixel_dist, int64_t *nrows, void **metadata);
extern "C" long _get_num_coords(long pixel_dist, long *p_nrows, long D,
                                void **metadata) {
  SWITCH_DIM(return, t_get_num_coords, pixel_dist, p_nrows, metadata)
}

template <uint8_t D>
long t_get_coords(long *coords, int64_t pixel_dist, void **metadata);
extern "C" long _get_coords(long *coords, long pixel_dist, long D,
                            void **metadata) {
  SWITCH_DIM(return, t_get_coords, coords, pixel_dist, metadata)
}

template <uint8_t D>
long t_get_permutation(long *p_permutation, int64_t pixel_dist_src,
                       int64_t pixel_dist_dst, void **metadata);
extern "C" long _get_permutation(long *p_permutation, long pixel_dist_src,
                                 long pixel_dist_dst, long D, void **metadata) {
  SWITCH_DIM(return, t_get_permutation, p_permutation, pixel_dist_src,
                   pixel_dist_dst, metadata)
}

template <uint8_t D> void t_clear(void **metadata);
extern "C" void _clear(long D, void **metadata) {
  SWITCH_DIM(NOARG, t_clear, metadata)
}

template <uint8_t D>
long t_conv_fw(const float *p_in_feat, int64_t in_nchannel, float *p_out_feat,
               int64_t out_nchannel, const float *p_kernel, int64_t out_nrows,
               int64_t pixel_dist, int64_t stride, int64_t kernel_size,
               int64_t dilation, int64_t region_type, int64_t *p_offset,
               int64_t n_offset, void **metadata);
extern "C" long _conv_fw(float *p_in_feat, long in_nchannel, float *p_out_feat,
                         long out_nchannel, float *p_kernel, long out_nrows,
                         long pixel_dist, long stride, long kernel_size,
                         long dilation, long region_type, int64_t *p_offset,
                         int64_t n_offset, long D, void **metadata) {
  SWITCH_DIM(return, t_conv_fw, p_in_feat, in_nchannel, p_out_feat,
                   out_nchannel, p_kernel, out_nrows, pixel_dist, stride,
                   kernel_size, dilation, region_type, p_offset, n_offset,
                   metadata)
}

template <uint8_t D>
long t_conv_tr_fw(const float *p_in_feat, int64_t in_nchannel,
                  float *p_out_feat, int64_t out_nchannel,
                  const float *p_kernel, int64_t out_nrows, int64_t pixel_dist,
                  int64_t stride, int64_t kernel_size, int64_t dilation,
                  int64_t region_type, int64_t *p_offset, int64_t n_offset,
                  void **metadata);
extern "C" long _conv_tr_fw(float *p_in_feat, long in_nchannel,
                            float *p_out_feat, long out_nchannel,
                            float *p_kernel, long out_nrows, long pixel_dist,
                            long stride, long kernel_size, long dilation,
                            long region_type, int64_t *p_offset,
                            int64_t n_offset, long D, void **metadata) {
  SWITCH_DIM(return, t_conv_tr_fw, p_in_feat, in_nchannel, p_out_feat,
                   out_nchannel, p_kernel, out_nrows, pixel_dist, stride,
                   kernel_size, dilation, region_type, p_offset, n_offset,
                   metadata)
}

template <uint8_t D>
long t_conv_bw(const float *p_in_feat, float *p_grad_in_feat,
               int64_t in_nchannel, float *p_grad_out_feat,
               int64_t out_nchannel, float *p_kernel, float *p_grad_kernel,
               int64_t out_nrows, int64_t pixel_dist, int64_t stride,
               int64_t kernel_size, int64_t dilation, void **metadata);
extern "C" long _conv_bw(float *p_in_feat, float *p_grad_in_feat,
                         long in_nchannel, float *p_grad_out_feat,
                         long out_nchannel, float *p_kernel,
                         float *p_grad_kernel, long out_nrows, long pixel_dist,
                         long stride, long kernel_size, long dilation, long D,
                         void **metadata) {
  SWITCH_DIM(return, t_conv_bw, p_in_feat, p_grad_in_feat, in_nchannel,
                   p_grad_out_feat, out_nchannel, p_kernel, p_grad_kernel,
                   out_nrows, pixel_dist, stride, kernel_size, dilation,
                   metadata)
}

template <uint8_t D>
long t_conv_tr_bw(const float *p_in_feat, float *p_grad_in_feat,
                  int64_t in_nchannel, float *p_grad_out_feat,
                  int64_t out_nchannel, float *p_kernel, float *p_grad_kernel,
                  int64_t out_nrows, int64_t pixel_dist, int64_t stride,
                  int64_t kernel_size, int64_t dilation, void **metadata);
extern "C" long _conv_tr_bw(float *p_in_feat, float *p_grad_in_feat,
                            long in_nchannel, float *p_grad_out_feat,
                            long out_nchannel, float *p_kernel,
                            float *p_grad_kernel, long out_nrows,
                            long pixel_dist, long stride, long kernel_size,
                            long dilation, long D, void **metadata) {
  SWITCH_DIM(return, t_conv_tr_bw, p_in_feat, p_grad_in_feat, in_nchannel,
                   p_grad_out_feat, out_nchannel, p_kernel, p_grad_kernel,
                   out_nrows, pixel_dist, stride, kernel_size, dilation,
                   metadata)
}

template <uint8_t D>
long t_conv_fw_gpu(const float *d_in_feat, int64_t in_nchannel,
                   float *d_out_feat, int64_t out_nchannel,
                   const float *d_kernel, int64_t out_nrows, int64_t pixel_dist,
                   int64_t stride, int64_t kernel_size, int64_t dilation,
                   int64_t region_type, int64_t *p_offset, int64_t n_offset,
                   cudaStream_t stream, void **metadata);
extern "C" long _conv_fw_gpu(float *d_in_feat, long in_nchannel,
                             float *d_out_feat, long out_nchannel,
                             float *d_kernel, long out_nrows, long pixel_dist,
                             long stride, long kernel_size, long dilation,
                             long region_type, long *p_offset, long n_offset,
                             cudaStream_t stream, long D, void **metadata) {
  SWITCH_DIM(return, t_conv_fw_gpu, d_in_feat, in_nchannel, d_out_feat,
                   out_nchannel, d_kernel, out_nrows, pixel_dist, stride,
                   kernel_size, dilation, region_type, p_offset, n_offset,
                   stream, metadata)
}

template <uint8_t D>
long t_conv_tr_fw_gpu(const float *d_in_feat, int64_t in_nchannel,
                      float *d_out_feat, int64_t out_nchannel,
                      const float *d_kernel, int64_t out_nrows,
                      int64_t pixel_dist, int64_t stride, int64_t kernel_size,
                      int64_t dilation, int64_t region_type, int64_t *p_offset,
                      int64_t n_offset, cudaStream_t stream, void **metadata);
extern "C" long _conv_tr_fw_gpu(float *d_in_feat, long in_nchannel,
                                float *d_out_feat, long out_nchannel,
                                float *d_kernel, long out_nrows,
                                long pixel_dist, long stride, long kernel_size,
                                long dilation, long region_type, long *p_offset,
                                long n_offset, cudaStream_t stream, long D,
                                void **metadata) {
  SWITCH_DIM(return, t_conv_tr_fw_gpu, d_in_feat, in_nchannel, d_out_feat,
                   out_nchannel, d_kernel, out_nrows, pixel_dist, stride,
                   kernel_size, dilation, region_type, p_offset, n_offset,
                   stream, metadata)
}

template <uint8_t D>
long t_conv_bw_gpu(const float *d_in_feat, float *d_grad_in_feat,
                   int64_t in_nchannel, float *d_grad_out_feat,
                   int64_t out_nchannel, float *d_kernel, float *d_grad_kernel,
                   int64_t out_nrows, int64_t pixel_dist, int64_t stride,
                   int64_t kernel_size, int64_t dilation, cudaStream_t stream,
                   void **metadata);
extern "C" long _conv_bw_gpu(float *d_in_feat, float *d_grad_in_feat,
                             long in_nchannel, float *d_grad_out_feat,
                             long out_nchannel, float *d_kernel,
                             float *d_grad_kernel, long out_nrows,
                             long pixel_dist, long stride, long kernel_size,
                             long dilation, cudaStream_t stream, long D,
                             void **metadata) {
  SWITCH_DIM(return, t_conv_bw_gpu, d_in_feat, d_grad_in_feat, in_nchannel,
                   d_grad_out_feat, out_nchannel, d_kernel, d_grad_kernel,
                   out_nrows, pixel_dist, stride, kernel_size, dilation, stream,
                   metadata);
}

template <uint8_t D>
long t_conv_tr_bw_gpu(const float *d_in_feat, float *d_grad_in_feat,
                      int64_t in_nchannel, float *d_grad_out_feat,
                      int64_t out_nchannel, float *d_kernel,
                      float *d_grad_kernel, int64_t out_nrows,
                      int64_t pixel_dist, int64_t stride, int64_t kernel_size,
                      int64_t dilation, cudaStream_t stream, void **metadata);
extern "C" long _conv_tr_bw_gpu(float *d_in_feat, float *d_grad_in_feat,
                                long in_nchannel, float *d_grad_out_feat,
                                long out_nchannel, float *d_kernel,
                                float *d_grad_kernel, long out_nrows,
                                long pixel_dist, long stride, long kernel_size,
                                long dilation, cudaStream_t stream, long D,
                                void **metadata) {
  SWITCH_DIM(return, t_conv_tr_bw_gpu, d_in_feat, d_grad_in_feat, in_nchannel,
                   d_grad_out_feat, out_nchannel, d_kernel, d_grad_kernel,
                   out_nrows, pixel_dist, stride, kernel_size, dilation, stream,
                   metadata);
}
#endif
