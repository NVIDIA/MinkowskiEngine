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

namespace minkowski {

#define MINK_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, HINT, ...)    \
  case enum_type: {                                                            \
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
