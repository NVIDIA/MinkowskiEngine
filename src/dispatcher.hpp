#pragma once

#include <ATen/core/DeprecatedTypeProperties.h>
#include <ATen/core/Tensor.h>
#include <ATen/record_function.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/complex.h>
#include <c10/util/string_view.h>

#ifdef XPLAT_MOBILE_BUILD
#include <ATen/selected_mobile_ops.h>
#else
namespace at {
/**
 * The method should_include_kernel_dtype() returns true/false
 * based on whether the switching code for a specific dtype should be
 * included based on build time constants generated from tracing model
 * execution. This method will be implmeneted via code-generation and
 * included in this file when code-gen is ready.
 */
inline constexpr bool should_include_kernel_dtype(const char *kernel_tag_str,
                                                  at::ScalarType scalar_type) {
  return true;
}
} // namespace at
#endif

namespace minkowski {

#define MINK_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, HINT, ...)    \
  case enum_type: {                                                            \
    at::guts::if_constexpr<(                                                   \
        !at::should_include_kernel_dtype(NAME, enum_type))>([&] {              \
      AT_ERROR("dtype '", toString(enum_type),                                 \
               "' not selected for kernel tag ", #NAME);                       \
    });                                                                        \
    using HINT = type;                                                         \
    return __VA_ARGS__();                                                      \
  }

#define MINK_PRIVATE_CASE_TYPE(NAME, enum_type, type, HINT, ...)               \
  MINK_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, HINT, __VA_ARGS__)

#define MINK_DISPATCH_INTEGER_TYPES(TYPE, HINT, NAME, ...)                     \
  [&] {                                                                        \
    const auto &the_type = TYPE;                                               \
    /* don't use TYPE again in case it is an expensive or side-effect op */    \
    at::ScalarType _it = ::detail::scalar_type(the_type);                      \
    switch (_it) {                                                             \
      MINK_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int32_t, HINT,         \
                             __VA_ARGS__)                                      \
      MINK_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Long, int64_t, HINT,        \
                             __VA_ARGS__)                                      \
    default:                                                                   \
      AT_ERROR(#NAME, " not implemented for '", toString(_it), "'");           \
    }                                                                          \
  }()

} // namespace minkowski
