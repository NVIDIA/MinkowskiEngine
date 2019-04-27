#include "region_iter.hpp"
#include "instantiation.hpp"

template <uint8_t D, typename Itype>
Region<D, Itype>::Region(const Coord<D, Itype> &center_,
                         const Arr<D, int> &tensor_strides,
                         const Arr<D, int> &kernel_size,
                         const Arr<D, int> &dilations, int region_type,
                         const Itype *p_offset, int n_offset)
    : region_type(region_type), tensor_strides(tensor_strides),
      kernel_size(kernel_size), dilations(dilations), p_offset(p_offset),
      n_offset(n_offset), use_lower_bound(false) {
  for (int i = 0; i < D; i++) {
    center[i] = center_[i];
    lb[i] =
        center_[i] - int(kernel_size[i] / 2) * dilations[i] * tensor_strides[i];
    ub[i] =
        center_[i] + int(kernel_size[i] / 2) * dilations[i] * tensor_strides[i];
  }
  lb[D] = ub[D] = center[D] = center_[D]; // set the batch index
}

template <uint8_t D, typename Itype>
Region<D, Itype>::Region(const Coord<D, Itype> &lower_bound_,
                         const Arr<D, int> &tensor_strides,
                         const Arr<D, int> &kernel_size,
                         const Arr<D, int> &dilations, int region_type,
                         const Itype *p_offset, int n_offset,
                         bool use_lower_bound)
    : region_type(region_type), tensor_strides(tensor_strides),
      kernel_size(kernel_size), dilations(dilations), p_offset(p_offset),
      n_offset(n_offset), use_lower_bound(true) {
  if (region_type > 0)
    throw std::invalid_argument(
        Formatter() << "The region type " << region_type
                    << " is not supported with the use_lower_bound argument");
  for (int i = 0; i < D; i++) {
    lb[i] = lower_bound_[i];
    ub[i] = lower_bound_[i] + kernel_size[i] * dilations[i] * tensor_strides[i];
  }
  lb[D] = ub[D] = lower_bound_[D]; // set the batch index
}

INSTANTIATE_CLASS_DIM_ITYPE(Region, int32_t);
