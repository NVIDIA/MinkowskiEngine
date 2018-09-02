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
  default:                                                                     \
    printf("%s\n", "Dimension mismatch");                                      \
  }

#define SWITCH_DIM_ITYPE(RETURN, func, Itype, ...)                             \
  switch (D) {                                                                 \
  case 2:                                                                      \
    RETURN func<2, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 3:                                                                      \
    RETURN func<3, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 4:                                                                      \
    RETURN func<4, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 5:                                                                      \
    RETURN func<5, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 6:                                                                      \
    RETURN func<6, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 7:                                                                      \
    RETURN func<7, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  default:                                                                     \
    printf("%s\n", "Dimension mismatch");                                      \
  }

#define SWITCH_DIM_TYPES(RETURN, func, Dtype, Itype, ...)                      \
  switch (D) {                                                                 \
  case 2:                                                                      \
    RETURN func<2, Dtype, Itype>(__VA_ARGS__);                                 \
    break;                                                                     \
  case 3:                                                                      \
    RETURN func<3, Dtype, Itype>(__VA_ARGS__);                                 \
    break;                                                                     \
  case 4:                                                                      \
    RETURN func<4, Dtype, Itype>(__VA_ARGS__);                                 \
    break;                                                                     \
  case 5:                                                                      \
    RETURN func<5, Dtype, Itype>(__VA_ARGS__);                                 \
    break;                                                                     \
  case 6:                                                                      \
    RETURN func<6, Dtype, Itype>(__VA_ARGS__);                                 \
    break;                                                                     \
  case 7:                                                                      \
    RETURN func<7, Dtype, Itype>(__VA_ARGS__);                                 \
    break;                                                                     \
  default:                                                                     \
    printf("%s\n", "Dimension mismatch");                                      \
  }

template <uint8_t D, typename Itype>
long t_initialize_coords(const Itype *coords, int nrows,
                         const Itype *p_pixel_dist, void **metadata);
extern "C" long _initialize_coords(int *coords, int nrows, int *p_pixel_dist,
                                   int D, void **metadata) {
  SWITCH_DIM_ITYPE(return, t_initialize_coords, int32_t, coords, nrows,
                         p_pixel_dist, metadata)
}

template <uint8_t D, typename Itype>
long t_initialize_coords_with_duplicates(const Itype *coords, int nrows,
                                         const Itype *p_pixel_dist,
                                         void **metadata);
extern "C" long _initialize_coords_with_duplicates(int *coords, int nrows,
                                                   int *p_pixel_dist, int D,
                                                   void **metadata) {
  SWITCH_DIM_ITYPE(return, t_initialize_coords_with_duplicates, int32_t, coords,
                         nrows, p_pixel_dist, metadata)
}

template <uint8_t D, typename Itype>
long t_initialize_out_coords(const Itype *p_pixel_dist, const Itype *p_stride,
                             bool is_transpose, void **metadata);
extern "C" long _initialize_out_coords(int *p_pixel_dist, int *p_stride,
                                       bool is_transpose, int D,
                                       void **metadata) {
  SWITCH_DIM_ITYPE(return, t_initialize_out_coords, int32_t, p_pixel_dist,
                         p_stride, is_transpose, metadata)
}

template <uint8_t D, typename Itype>
long t_initialize_origin_coords(const Itype *p_pixel_dist, int batch_size,
                                void **metadata);
extern "C" long _initialize_origin_coords(int *p_pixel_dist, int batch_size,
                                          int D, void **metadata) {
  SWITCH_DIM_ITYPE(return, t_initialize_origin_coords, int32_t, p_pixel_dist,
                         batch_size, metadata)
}

template <uint8_t D, typename Itype>
long t_get_index_map(const Itype *coords, int nrows, Itype *p_index_map,
                     int index_map_nrows, const Itype *p_pixel_dist,
                     void **metadata);
extern "C" long _get_index_map(int *coords, int nrows, int *p_index_map,
                               int index_map_nrows, int *p_pixel_dist, int D,
                               void **metadata) {
  SWITCH_DIM_ITYPE(return, t_get_index_map, int32_t, coords, nrows, p_index_map,
                         index_map_nrows, p_pixel_dist, metadata)
}

template <uint8_t D, typename Itype>
long t_get_num_coords(const Itype *p_pixel_dist, int *nrows, void **metadata);
extern "C" long _get_num_coords(int *p_pixel_dist, int *p_nrows, int D,
                                void **metadata) {
  SWITCH_DIM_ITYPE(return, t_get_num_coords, int32_t, p_pixel_dist, p_nrows,
                         metadata)
}

template <uint8_t D, typename Itype>
long t_get_coords(Itype *coords, const Itype *p_pixel_dist, void **metadata);
extern "C" long _get_coords(int *coords, int *p_pixel_dist, int D,
                            void **metadata) {
  SWITCH_DIM_ITYPE(return, t_get_coords, int32_t, coords, p_pixel_dist,
                         metadata)
}

template <uint8_t D, typename Itype>
long t_get_permutation(Itype *p_permutation, const Itype *p_pixel_dist_src,
                       const Itype *p_pixel_dist_dst, void **metadata);
extern "C" long _get_permutation(int *p_permutation, int *p_pixel_dist_src,
                                 int *p_pixel_dist_dst, int D,
                                 void **metadata) {
  SWITCH_DIM_ITYPE(return, t_get_permutation, int32_t, p_permutation,
                         p_pixel_dist_src, p_pixel_dist_dst, metadata)
}

template <uint8_t D, typename Itype> void t_clear(void **metadata);
extern "C" void _clear(int D, void **metadata) {
  SWITCH_DIM_ITYPE(NOARG, t_clear, int32_t, metadata)
}

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_fw(const Dtype *p_in_feat, Itype in_nchannel, Dtype *p_out_feat,
               Itype out_nchannel, const Dtype *p_kernel, Itype out_nrows,
               const Itype *p_pixel_dist, const Itype *p_stride,
               const Itype *p_kernel_size, const Itype *p_dilation,
               Itype region_type, const Itype *p_offset, Itype n_offset,
               uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
               void **metadata);
extern "C" long _conv_fw(float *p_in_feat, int in_nchannel, float *p_out_feat,
                         int out_nchannel, float *p_kernel, int out_nrows,
                         int *p_pixel_dist, int *p_stride, int *p_kernel_size,
                         int *p_dilation, int region_type, int *p_offset,
                         int n_offset, uint64_t *p_in_coords_key,
                         uint64_t *p_out_coords_key, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_conv_fw, float, int32_t, p_in_feat, in_nchannel,
                         p_out_feat, out_nchannel, p_kernel, out_nrows,
                         p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                         region_type, p_offset, n_offset, p_in_coords_key,
                         p_out_coords_key, metadata)
}

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_bw(const Dtype *p_in_feat, Dtype *p_grad_in_feat, Itype in_nchannel,
               const Dtype *p_grad_out_feat, Itype out_nchannel,
               const Dtype *p_kernel, Dtype *p_grad_kernel, Itype out_nrows,
               const Itype *p_pixel_dist, const Itype *p_stride,
               const Itype *p_kernel_size, const Itype *p_dilation,
               uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
               void **metadata);
extern "C" long _conv_bw(float *p_in_feat, float *p_grad_in_feat,
                         int in_nchannel, float *p_grad_out_feat,
                         int out_nchannel, float *p_kernel,
                         float *p_grad_kernel, int out_nrows, int *p_pixel_dist,
                         int *p_stride, int *p_kernel_size, int *p_dilation,
                         uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                         int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_conv_bw, float, int32_t, p_in_feat, p_grad_in_feat,
                         in_nchannel, p_grad_out_feat, out_nchannel, p_kernel,
                         p_grad_kernel, out_nrows, p_pixel_dist, p_stride,
                         p_kernel_size, p_dilation, p_in_coords_key,
                         p_out_coords_key, metadata)
}

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_tr_fw(const Dtype *p_in_feat, Itype in_nchannel, Dtype *p_out_feat,
                  Itype out_nchannel, const Dtype *p_kernel, Itype out_nrows,
                  const Itype *p_pixel_dist, const Itype *p_stride,
                  const Itype *p_kernel_size, const Itype *p_dilation,
                  Itype region_type, const Itype *p_offset, Itype n_offset,
                  uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                  void **metadata);
extern "C" long
_conv_tr_fw(float *p_in_feat, int in_nchannel, float *p_out_feat,
            int out_nchannel, float *p_kernel, int out_nrows, int *p_pixel_dist,
            int *p_stride, int *p_kernel_size, int *p_dilation, int region_type,
            int *p_offset, int n_offset, uint64_t *p_in_coords_key,
            uint64_t *p_out_coords_key, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_conv_tr_fw, float, int32_t, p_in_feat, in_nchannel,
                         p_out_feat, out_nchannel, p_kernel, out_nrows,
                         p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                         region_type, p_offset, n_offset, p_in_coords_key,
                         p_out_coords_key, metadata)
}

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_tr_bw(const Dtype *p_in_feat, Dtype *p_grad_in_feat,
                  Itype in_nchannel, const Dtype *p_grad_out_feat,
                  Itype out_nchannel, const Dtype *p_kernel,
                  Dtype *p_grad_kernel, Itype out_nrows,
                  const Itype *p_pixel_dist, const Itype *p_stride,
                  const Itype *p_kernel_size, const Itype *p_dilation,
                  uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                  void **metadata);
extern "C" long
_conv_tr_bw(float *p_in_feat, float *p_grad_in_feat, int in_nchannel,
            float *p_grad_out_feat, int out_nchannel, float *p_kernel,
            float *p_grad_kernel, int out_nrows, int *p_pixel_dist,
            int *p_stride, int *p_kernel_size, int *p_dilation,
            uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, int D,
            void **metadata) {
  SWITCH_DIM_TYPES(return, t_conv_tr_bw, float, int32_t, p_in_feat,
                         p_grad_in_feat, in_nchannel, p_grad_out_feat,
                         out_nchannel, p_kernel, p_grad_kernel, out_nrows,
                         p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                         p_in_coords_key, p_out_coords_key, metadata)
}

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_fw_gpu(const Dtype *d_in_feat, Itype in_nchannel, Dtype *d_out_feat,
                   Itype out_nchannel, const Dtype *d_kernel, Itype out_nrows,
                   const Itype *p_pixel_dist, const Itype *p_stride,
                   const Itype *p_kernel_size, const Itype *p_dilation,
                   Itype region_type, const Itype *p_offset, Itype n_offset,
                   uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                   cudaStream_t stream, void **metadata);
extern "C" long _conv_fw_gpu(float *d_in_feat, int in_nchannel,
                             float *d_out_feat, int out_nchannel,
                             float *d_kernel, int out_nrows, int *p_pixel_dist,
                             int *p_stride, int *p_kernel_size, int *p_dilation,
                             int region_type, int *p_offset, int n_offset,
                             uint64_t *p_in_coords_key,
                             uint64_t *p_out_coords_key, cudaStream_t stream,
                             int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_conv_fw_gpu, float, int32_t, d_in_feat,
                         in_nchannel, d_out_feat, out_nchannel, d_kernel,
                         out_nrows, p_pixel_dist, p_stride, p_kernel_size,
                         p_dilation, region_type, p_offset, n_offset,
                         p_in_coords_key, p_out_coords_key, stream, metadata)
}

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_tr_fw_gpu(const Dtype *d_in_feat, Itype in_nchannel,
                      Dtype *d_out_feat, Itype out_nchannel,
                      const Dtype *d_kernel, Itype out_nrows,
                      const Itype *p_pixel_dist, const Itype *p_stride,
                      const Itype *p_kernel_size, const Itype *p_dilation,
                      Itype region_type, const Itype *p_offset, Itype n_offset,
                      uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                      cudaStream_t stream, void **metadata);
extern "C" long
_conv_tr_fw_gpu(float *d_in_feat, int in_nchannel, float *d_out_feat,
                int out_nchannel, float *d_kernel, int out_nrows,
                int *p_pixel_dist, int *p_stride, int *p_kernel_size,
                int *p_dilation, int region_type, int *p_offset, int n_offset,
                uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                cudaStream_t stream, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_conv_tr_fw_gpu, float, int32_t, d_in_feat,
                         in_nchannel, d_out_feat, out_nchannel, d_kernel,
                         out_nrows, p_pixel_dist, p_stride, p_kernel_size,
                         p_dilation, region_type, p_offset, n_offset,
                         p_in_coords_key, p_out_coords_key, stream, metadata)
}

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_bw_gpu(const Dtype *d_in_feat, Dtype *d_grad_in_feat,
                   Itype in_nchannel, const Dtype *d_grad_out_feat,
                   Itype out_nchannel, const Dtype *d_kernel,
                   Dtype *d_grad_kernel, Itype out_nrows,
                   const Itype *p_pixel_dist, const Itype *p_stride,
                   const Itype *p_kernel_size, const Itype *p_dilation,
                   uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                   cudaStream_t stream, void **metadata);
extern "C" long
_conv_bw_gpu(float *d_in_feat, float *d_grad_in_feat, int in_nchannel,
             float *d_grad_out_feat, int out_nchannel, float *d_kernel,
             float *d_grad_kernel, int out_nrows, int *p_pixel_dist,
             int *p_stride, int *p_kernel_size, int *p_dilation,
             uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
             cudaStream_t stream, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_conv_bw_gpu, float, int32_t, d_in_feat,
                         d_grad_in_feat, in_nchannel, d_grad_out_feat,
                         out_nchannel, d_kernel, d_grad_kernel, out_nrows,
                         p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                         p_in_coords_key, p_out_coords_key, stream, metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_conv_tr_bw_gpu(const Dtype *d_in_feat, Dtype *d_grad_in_feat,
                      Itype in_nchannel, const Dtype *d_grad_out_feat,
                      Itype out_nchannel, const Dtype *d_kernel,
                      Dtype *d_grad_kernel, Itype out_nrows,
                      const Itype *p_pixel_dist, const Itype *p_stride,
                      const Itype *p_kernel_size, const Itype *p_dilation,
                      uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                      cudaStream_t stream, void **metadata);
extern "C" long
_conv_tr_bw_gpu(float *d_in_feat, float *d_grad_in_feat, int in_nchannel,
                float *d_grad_out_feat, int out_nchannel, float *d_kernel,
                float *d_grad_kernel, int out_nrows, int *p_pixel_dist,
                int *p_stride, int *p_kernel_size, int *p_dilation,
                uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                cudaStream_t stream, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_conv_tr_bw_gpu, float, int32_t, d_in_feat,
                         d_grad_in_feat, in_nchannel, d_grad_out_feat,
                         out_nchannel, d_kernel, d_grad_kernel, out_nrows,
                         p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                         p_in_coords_key, p_out_coords_key, stream, metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_max_pooling_fw(const Dtype *p_in_feat, Dtype *p_out_feat,
                      Itype *p_mask_index, Itype nchannel, Itype out_nrows,
                      const Itype *p_pixel_dist, const Itype *p_stride,
                      const Itype *p_kernel_size, const Itype *p_dilation,
                      Itype region_type, const Itype *p_offset, Itype n_offset,
                      uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                      void **metadata);
extern "C" long
_max_pooling_fw(float *p_in_feat, float *p_out_feat, int *p_mask_index,
                int nchannel, int out_nrows, int *p_pixel_dist, int *p_stride,
                int *p_kernel_size, int *p_dilation, int region_type,
                int *p_offset, int n_offset, uint64_t *p_in_coords_key,
                uint64_t *p_out_coords_key, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_max_pooling_fw, float, int32_t, p_in_feat,
                         p_out_feat, p_mask_index, nchannel, out_nrows,
                         p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                         region_type, p_offset, n_offset, p_in_coords_key,
                         p_out_coords_key, metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_max_pooling_bw(Dtype *p_grad_in_feat, Itype in_nrows,
                      Dtype *p_grad_out_feat, Itype out_nrows,
                      const Itype *p_mask_index, Itype nchannel,
                      const Itype *p_pixel_dist, const Itype *p_stride,
                      const Itype *p_kernel_size, const Itype *p_dilation,
                      uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                      void **metadata);
extern "C" long
_max_pooling_bw(float *p_grad_in_feat, int in_nrows, float *p_grad_out_feat,
                int out_nrows, int *p_mask_index, int nchannel,
                int *p_pixel_dist, int *p_stride, int *p_kernel_size,
                int *p_dilation, uint64_t *p_in_coords_key,
                uint64_t *p_out_coords_key, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_max_pooling_bw, float, int32_t, p_grad_in_feat,
                         in_nrows, p_grad_out_feat, out_nrows, p_mask_index,
                         nchannel, p_pixel_dist, p_stride, p_kernel_size,
                         p_dilation, p_in_coords_key, p_out_coords_key,
                         metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_max_pooling_fw_gpu(const Dtype *d_in_feat, Dtype *d_out_feat,
                          Itype out_nrows, Itype *d_mask_index, Itype nchannel,
                          const Itype *p_pixel_dist, const Itype *p_stride,
                          const Itype *p_kernel_size, const Itype *p_dilation,
                          Itype region_type, const Itype *p_offset,
                          Itype n_offset, uint64_t *p_in_coords_key,
                          uint64_t *p_out_coords_key, cudaStream_t stream,
                          void **metadata);
extern "C" long
_max_pooling_fw_gpu(float *d_in_feat, float *d_out_feat, int out_nrows,
                    int *d_mask_index, int nchannel, int *p_pixel_dist,
                    int *p_stride, int *p_kernel_size, int *p_dilation,
                    int region_type, int *p_offset, int n_offset,
                    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                    cudaStream_t stream, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_max_pooling_fw_gpu, float, int32_t, d_in_feat,
                         d_out_feat, out_nrows, d_mask_index, nchannel,
                         p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                         region_type, p_offset, n_offset, p_in_coords_key,
                         p_out_coords_key, stream, metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_max_pooling_bw_gpu(Dtype *d_grad_in_feat, Itype in_nrows,
                          const Dtype *d_grad_out_feat, Itype out_nrows,
                          const Itype *d_mask_index, Itype nchannel,
                          const Itype *p_pixel_dist, const Itype *p_stride,
                          const Itype *p_kernel_size, const Itype *p_dilation,
                          uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                          cudaStream_t stream, void **metadata);
extern "C" long _max_pooling_bw_gpu(
    float *d_grad_in_feat, int in_nrows, float *d_grad_out_feat, int out_nrows,
    int *d_mask_index, int nchannel, int *p_pixel_dist, int *p_stride,
    int *p_kernel_size, int *p_dilation, uint64_t *p_in_coords_key,
    uint64_t *p_out_coords_key, cudaStream_t stream, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_max_pooling_bw_gpu, float, int32_t, d_grad_in_feat,
                         in_nrows, d_grad_out_feat, out_nrows, d_mask_index,
                         nchannel, p_pixel_dist, p_stride, p_kernel_size,
                         p_dilation, p_in_coords_key, p_out_coords_key, stream,
                         metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_nonzero_avg_pooling_fw(const Dtype *p_in_feat, Dtype *p_out_feat,
                              Dtype *p_num_nonzero, Itype nchannel,
                              Itype out_nrows, const Itype *p_pixel_dist,
                              const Itype *p_stride, const Itype *p_kernel_size,
                              const Itype *p_dilation, Itype region_type,
                              const Itype *p_offset, Itype n_offset,
                              uint64_t *p_in_coords_key,
                              uint64_t *p_out_coords_key, void **metadata);
extern "C" long
_nonzero_avg_pooling_fw(float *p_in_feat, float *p_out_feat,
                        float *p_num_nonzero, int nchannel, int out_nrows,
                        int *p_pixel_dist, int *p_stride, int *p_kernel_size,
                        int *p_dilation, int region_type, int *p_offset,
                        int n_offset, uint64_t *p_in_coords_key,
                        uint64_t *p_out_coords_key, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_nonzero_avg_pooling_fw, float, int32_t, p_in_feat,
                         p_out_feat, p_num_nonzero, nchannel, out_nrows,
                         p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                         region_type, p_offset, n_offset, p_in_coords_key,
                         p_out_coords_key, metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_nonzero_avg_pooling_bw(Dtype *p_grad_in_feat, Itype in_nrows,
                              Dtype *p_grad_out_feat, Itype out_nrows,
                              const Dtype *p_num_nonzero, Itype nchannel,
                              const Itype *p_pixel_dist, const Itype *p_stride,
                              const Itype *p_kernel_size,
                              const Itype *p_dilation,
                              uint64_t *p_in_coords_key,
                              uint64_t *p_out_coords_key, void **metadata);
extern "C" long _nonzero_avg_pooling_bw(
    float *p_grad_in_feat, int in_nrows, float *p_grad_out_feat, int out_nrows,
    float *p_num_nonzero, int nchannel, int *p_pixel_dist, int *p_stride,
    int *p_kernel_size, int *p_dilation, uint64_t *p_in_coords_key,
    uint64_t *p_out_coords_key, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_nonzero_avg_pooling_bw, float, int32_t,
                         p_grad_in_feat, in_nrows, p_grad_out_feat, out_nrows,
                         p_num_nonzero, nchannel, p_pixel_dist, p_stride,
                         p_kernel_size, p_dilation, p_in_coords_key,
                         p_out_coords_key, metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_nonzero_avg_pooling_fw_gpu(
    const Dtype *d_in_feat, Itype in_nrows, Dtype *d_out_feat, Itype out_nrows,
    Dtype *d_num_nonzero, Itype nchannel, const Itype *p_pixel_dist,
    const Itype *p_stride, const Itype *p_kernel_size, const Itype *p_dilation,
    Itype region_type, const Itype *p_offset, Itype n_offset,
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, cudaStream_t stream,
    void **metadata);
extern "C" long _nonzero_avg_pooling_fw_gpu(
    float *d_in_feat, int in_nrows, float *d_out_feat, int out_nrows,
    float *d_num_nonzero, int nchannel, int *p_pixel_dist, int *p_stride,
    int *p_kernel_size, int *p_dilation, int region_type, int *p_offset,
    int n_offset, uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
    cudaStream_t stream, int D, void **metadata) {
  SWITCH_DIM_TYPES(
      return, t_nonzero_avg_pooling_fw_gpu, float, int32_t, d_in_feat, in_nrows,
            d_out_feat, out_nrows, d_num_nonzero, nchannel, p_pixel_dist,
            p_stride, p_kernel_size, p_dilation, region_type, p_offset,
            n_offset, p_in_coords_key, p_out_coords_key, stream, metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_nonzero_avg_pooling_bw_gpu(
    Dtype *d_grad_in_feat, Itype in_nrows, const Dtype *d_grad_out_feat,
    Itype out_nrows, const Dtype *d_num_nonzero, Itype nchannel,
    const Itype *p_pixel_dist, const Itype *p_stride,
    const Itype *p_kernel_size, const Itype *p_dilation,
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, cudaStream_t stream,
    void **metadata);
extern "C" long _nonzero_avg_pooling_bw_gpu(
    float *d_grad_in_feat, int in_nrows, float *d_grad_out_feat, int out_nrows,
    float *d_num_nonzero, int nchannel, int *p_pixel_dist, int *p_stride,
    int *p_kernel_size, int *p_dilation, uint64_t *p_in_coords_key,
    uint64_t *p_out_coords_key, cudaStream_t stream, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_nonzero_avg_pooling_bw_gpu, float, int32_t,
                         d_grad_in_feat, in_nrows, d_grad_out_feat, out_nrows,
                         d_num_nonzero, nchannel, p_pixel_dist, p_stride,
                         p_kernel_size, p_dilation, p_in_coords_key,
                         p_out_coords_key, stream, metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_unpooling_fw(const Dtype *p_in_feat, Dtype *p_out_feat,
                    Dtype *p_num_nonzero, Itype nchannel, Itype out_nrows,
                    const Itype *p_pixel_dist, const Itype *p_stride,
                    const Itype *p_kernel_size, const Itype *p_dilation,
                    Itype region_type, const Itype *p_offset, Itype n_offset,
                    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                    void **metadata);
extern "C" long
_unpooling_fw(float *p_in_feat, float *p_out_feat, float *p_num_nonzero,
              int nchannel, int out_nrows, int *p_pixel_dist, int *p_stride,
              int *p_kernel_size, int *p_dilation, int region_type,
              int *p_offset, int n_offset, uint64_t *p_in_coords_key,
              uint64_t *p_out_coords_key, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_unpooling_fw, float, int32_t, p_in_feat,
                         p_out_feat, p_num_nonzero, nchannel, out_nrows,
                         p_pixel_dist, p_stride, p_kernel_size, p_dilation,
                         region_type, p_offset, n_offset, p_in_coords_key,
                         p_out_coords_key, metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_unpooling_bw(Dtype *p_grad_in_feat, Itype in_nrows,
                    Dtype *p_grad_out_feat, Itype out_nrows,
                    const Dtype *p_num_nonzero, Itype nchannel,
                    const Itype *p_pixel_dist, const Itype *p_stride,
                    const Itype *p_kernel_size, const Itype *p_dilation,
                    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                    void **metadata);
extern "C" long
_unpooling_bw(float *p_grad_in_feat, int in_nrows, float *p_grad_out_feat,
              int out_nrows, float *p_num_nonzero, int nchannel,
              int *p_pixel_dist, int *p_stride, int *p_kernel_size,
              int *p_dilation, uint64_t *p_in_coords_key,
              uint64_t *p_out_coords_key, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_unpooling_bw, float, int32_t, p_grad_in_feat,
                         in_nrows, p_grad_out_feat, out_nrows, p_num_nonzero,
                         nchannel, p_pixel_dist, p_stride, p_kernel_size,
                         p_dilation, p_in_coords_key, p_out_coords_key,
                         metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_unpooling_fw_gpu(const Dtype *d_in_feat, Itype in_nrows,
                        Dtype *d_out_feat, Itype out_nrows,
                        Dtype *d_num_nonzero, Itype nchannel,
                        const Itype *p_pixel_dist, const Itype *p_stride,
                        const Itype *p_kernel_size, const Itype *p_dilation,
                        Itype region_type, const Itype *p_offset,
                        Itype n_offset, uint64_t *p_in_coords_key,
                        uint64_t *p_out_coords_key, cudaStream_t stream,
                        void **metadata);
extern "C" long
_unpooling_fw_gpu(float *d_in_feat, int in_nrows, float *d_out_feat,
                  int out_nrows, float *d_num_nonzero, int nchannel,
                  int *p_pixel_dist, int *p_stride, int *p_kernel_size,
                  int *p_dilation, int region_type, int *p_offset, int n_offset,
                  uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                  cudaStream_t stream, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_unpooling_fw_gpu, float, int32_t, d_in_feat,
                         in_nrows, d_out_feat, out_nrows, d_num_nonzero,
                         nchannel, p_pixel_dist, p_stride, p_kernel_size,
                         p_dilation, region_type, p_offset, n_offset,
                         p_in_coords_key, p_out_coords_key, stream, metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_unpooling_bw_gpu(Dtype *d_grad_in_feat, Itype in_nrows,
                        const Dtype *d_grad_out_feat, Itype out_nrows,
                        const Dtype *d_num_nonzero, Itype nchannel,
                        const Itype *p_pixel_dist, const Itype *p_stride,
                        const Itype *p_kernel_size, const Itype *p_dilation,
                        uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                        cudaStream_t stream, void **metadata);
extern "C" long _unpooling_bw_gpu(
    float *d_grad_in_feat, int in_nrows, float *d_grad_out_feat, int out_nrows,
    float *d_num_nonzero, int nchannel, int *p_pixel_dist, int *p_stride,
    int *p_kernel_size, int *p_dilation, uint64_t *p_in_coords_key,
    uint64_t *p_out_coords_key, cudaStream_t stream, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_unpooling_bw_gpu, float, int32_t, d_grad_in_feat,
                         in_nrows, d_grad_out_feat, out_nrows, d_num_nonzero,
                         nchannel, p_pixel_dist, p_stride, p_kernel_size,
                         p_dilation, p_in_coords_key, p_out_coords_key, stream,
                         metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_global_avg_pooling_fw(const Dtype *p_in_feat, Dtype *p_out_feat,
                             Itype out_nrows, Itype nchannel,
                             Dtype *p_num_nonzero, const Itype *p_pixel_dist,
                             uint64_t *p_in_coords_key,
                             uint64_t *p_out_coords_key, void **metadata);
extern "C" long _global_avg_pooling_fw(float *p_in_feat, float *p_out_feat,
                                       int out_nrows, int nchannel,
                                       float *p_num_nonzero, int *p_pixel_dist,
                                       uint64_t *p_in_coords_key,
                                       uint64_t *p_out_coords_key, int D,
                                       void **metadata) {
  SWITCH_DIM_TYPES(return, t_global_avg_pooling_fw, float, int32_t, p_in_feat,
                         p_out_feat, out_nrows, nchannel, p_num_nonzero,
                         p_pixel_dist, p_in_coords_key, p_out_coords_key,
                         metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_global_avg_pooling_bw(Dtype *p_grad_in_feat, Itype in_nrows,
                             Dtype *p_grad_out_feat, Itype out_nrows,
                             Itype nchannel, const Dtype *p_num_nonzero,
                             const Itype *p_pixel_dist,
                             uint64_t *p_in_coords_key,
                             uint64_t *p_out_coords_key, void **metadata);
extern "C" long _global_avg_pooling_bw(float *p_grad_in_feat, int in_nrows,
                                       float *p_grad_out_feat, int out_nrows,
                                       int nchannel, float *p_num_nonzero,
                                       int *p_pixel_dist,
                                       uint64_t *p_in_coords_key,
                                       uint64_t *p_out_coords_key, int D,
                                       void **metadata) {
  SWITCH_DIM_TYPES(return, t_global_avg_pooling_bw, float, int32_t,
                         p_grad_in_feat, in_nrows, p_grad_out_feat, out_nrows,
                         nchannel, p_num_nonzero, p_pixel_dist, p_in_coords_key,
                         p_out_coords_key, metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_global_avg_pooling_fw_gpu(const Dtype *d_in_feat, Itype in_nrows,
                                 Dtype *d_out_feat, Itype out_nrows,
                                 Itype nchannel, Dtype *d_num_nonzero,
                                 const Itype *p_pixel_dist,
                                 uint64_t *p_in_coords_key,
                                 uint64_t *p_out_coords_key,
                                 cudaStream_t stream, void **metadata);
extern "C" long
_global_avg_pooling_fw_gpu(float *d_in_feat, int in_nrows, float *d_out_feat,
                           int out_nrows, int nchannel, float *d_num_nonzero,
                           int *p_pixel_dist, uint64_t *p_in_coords_key,
                           uint64_t *p_out_coords_key, cudaStream_t stream,
                           int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_global_avg_pooling_fw_gpu, float, int32_t,
                         d_in_feat, in_nrows, d_out_feat, out_nrows, nchannel,
                         d_num_nonzero, p_pixel_dist, p_in_coords_key,
                         p_out_coords_key, stream, metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_global_avg_pooling_bw_gpu(Dtype *d_grad_in_feat, Itype in_nrows,
                                 const Dtype *d_grad_out_feat, Itype out_nrows,
                                 Itype nchannel, const Dtype *d_num_nonzero,
                                 const Itype *p_pixel_dist,
                                 uint64_t *p_in_coords_key,
                                 uint64_t *p_out_coords_key,
                                 cudaStream_t stream, void **metadata);
extern "C" long _global_avg_pooling_bw_gpu(
    float *d_grad_in_feat, int in_nrows, float *d_grad_out_feat, int out_nrows,
    int nchannel, float *d_num_nonzero, int *p_pixel_dist,
    uint64_t *p_in_coords_key, uint64_t *p_out_coords_key, cudaStream_t stream,
    int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_global_avg_pooling_bw_gpu, float, int32_t,
                         d_grad_in_feat, in_nrows, d_grad_out_feat, out_nrows,
                         nchannel, d_num_nonzero, p_pixel_dist, p_in_coords_key,
                         p_out_coords_key, stream, metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_global_broadcast_fw(const Dtype *p_in_feat, int in_nrows,
                           const Dtype *p_in_feat_global, int in_nrows_global,
                           Dtype *p_out_feat, int nchannel,
                           const Itype *p_pixel_dist, int op,
                           uint64_t *p_in_coords_key,
                           uint64_t *p_out_coords_key, void **metadata);
extern "C" long
_global_broadcast_fw(float *p_in_feat, int in_nrows, float *p_in_feat_global,
                     int in_nrows_global, float *p_out_feat, int nchannel,
                     int *p_pixel_dist, int op, uint64_t *p_in_coords_key,
                     uint64_t *p_out_coords_key, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_global_broadcast_fw, float, int32_t, p_in_feat,
                         in_nrows, p_in_feat_global, in_nrows_global,
                         p_out_feat, nchannel, p_pixel_dist, op,
                         p_in_coords_key, p_out_coords_key, metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_global_broadcast_bw(const Dtype *p_in_feat, Dtype *p_grad_in_feat,
                           int in_nrows, const Dtype *p_in_feat_global,
                           Dtype *p_grad_in_feat_global, int in_nrows_global,
                           const Dtype *p_grad_out_feat, int nchannel,
                           const Itype *p_pixel_dist, int op,
                           uint64_t *p_in_coords_key,
                           uint64_t *p_out_coords_key, void **metadata);
extern "C" long
_global_broadcast_bw(float *p_in_feat, float *p_grad_in_feat, int in_nrows,
                     float *p_in_feat_global, float *p_grad_in_feat_global,
                     int in_nrows_global, float *p_grad_out_feat, int nchannel,
                     int *p_pixel_dist, int op, uint64_t *p_in_coords_key,
                     uint64_t *p_out_coords_key, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_global_broadcast_bw, float, int32_t, p_in_feat,
                         p_grad_in_feat, in_nrows, p_in_feat_global,
                         p_grad_in_feat_global, in_nrows_global,
                         p_grad_out_feat, nchannel, p_pixel_dist,
                         op, p_in_coords_key, p_out_coords_key, metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_global_broadcast_fw_gpu(const Dtype *p_in_feat, int in_nrows,
                               const Dtype *p_in_feat_global,
                               int in_nrows_global, Dtype *p_out_feat,
                               int nchannel, const Itype *p_pixel_dist, int op,
                               uint64_t *p_in_coords_key,
                               uint64_t *p_out_coords_key, cudaStream_t stream,
                               void **metadata);
extern "C" long _global_broadcast_fw_gpu(
    float *p_in_feat, int in_nrows, float *p_in_feat_global,
    int in_nrows_global, float *p_out_feat, int nchannel, int *p_pixel_dist,
    int op, uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
    cudaStream_t stream, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_global_broadcast_fw_gpu, float, int32_t, p_in_feat,
                         in_nrows, p_in_feat_global, in_nrows_global,
                         p_out_feat, nchannel, p_pixel_dist, op,
                         p_in_coords_key, p_out_coords_key, stream, metadata);
}

template <uint8_t D, typename Dtype, typename Itype>
long t_global_broadcast_bw_gpu(
    const Dtype *p_in_feat, Dtype *p_grad_in_feat, int in_nrows,
    const Dtype *p_in_feat_global, Dtype *p_grad_in_feat_global,
    int in_nrows_global, const Dtype *p_grad_out_feat, int nchannel,
    const Itype *p_pixel_dist, int op, uint64_t *p_in_coords_key,
    uint64_t *p_out_coords_key, cudaStream_t stream, void **metadata);
extern "C" long
_global_broadcast_bw_gpu(float *p_in_feat, float *p_grad_in_feat, int in_nrows,
                         float *p_in_feat_global, float *p_grad_in_feat_global,
                         int in_nrows_global, float *p_grad_out_feat,
                         int nchannel, int *p_pixel_dist, int op,
                         uint64_t *p_in_coords_key, uint64_t *p_out_coords_key,
                         cudaStream_t stream, int D, void **metadata) {
  SWITCH_DIM_TYPES(return, t_global_broadcast_bw_gpu, float, int32_t, p_in_feat,
                         p_grad_in_feat, in_nrows, p_in_feat_global,
                         p_grad_in_feat_global, in_nrows_global,
                         p_grad_out_feat, nchannel, p_pixel_dist, op,
                         p_in_coords_key, p_out_coords_key, stream, metadata);
}

#endif
