nanobind_add_module(
  vhacd_ext
  NB_STATIC # static libnanobind
  vhacd.cpp
)

install(TARGETS vhacd_ext LIBRARY DESTINATION .)