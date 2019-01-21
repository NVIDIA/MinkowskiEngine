#ifndef INSTANTIATION
#define INSTANTIATION

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname)                                           \
  char gInstantiationGuard##classname;                                         \
  template class classname<float>;                                             \
  // template class classname<double>
  //
#define INSTANTIATE_CLASS_DIM(CLASSNAME)                                       \
  template class CLASSNAME<1>;                                                 \
  template class CLASSNAME<2>;                                                 \
  template class CLASSNAME<3>;                                                 \
  template class CLASSNAME<4>;                                                 \
  template class CLASSNAME<5>;                                                 \
  template class CLASSNAME<6>;                                                 \
  template class CLASSNAME<7>;

#define INSTANTIATE_CLASS_DIM_ITYPE(CLASSNAME, ITYPE)                          \
  template class CLASSNAME<1, ITYPE>;                                          \
  template class CLASSNAME<2, ITYPE>;                                          \
  template class CLASSNAME<3, ITYPE>;                                          \
  template class CLASSNAME<4, ITYPE>;                                          \
  template class CLASSNAME<5, ITYPE>;                                          \
  template class CLASSNAME<6, ITYPE>;                                          \
  template class CLASSNAME<7, ITYPE>;

#define SWITCH_DIM_TYPES(func, Dtype, Itype, ...)                              \
  switch (D) {                                                                 \
  case 1:                                                                      \
    func<1, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 2:                                                                      \
    func<2, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 3:                                                                      \
    func<3, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 4:                                                                      \
    func<4, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 5:                                                                      \
    func<5, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 6:                                                                      \
    func<6, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 7:                                                                      \
    func<7, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  default:                                                                     \
    throw std::invalid_argument(Formatter() << "Not supported D " << D);       \
  }

#endif
