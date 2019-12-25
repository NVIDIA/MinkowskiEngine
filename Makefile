###############################################################################
# Uncomment for debugging
# DEBUG := 1
# Pretty build
Q ?= @

# Uncomment for CPU only build. From the command line, `python setup.py install --cpu_only`
# CPU_ONLY := 1

CXX := g++

EXTENSION_NAME := minkowski

# BLAS choice:
# atlas for ATLAS (default)
# mkl for MKL
# open for OpenBlas
BLAS := open

# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
# Leave commented to accept the defaults for your choice of BLAS
# (which should work)!
# BLAS_INCLUDE := /path/to/your/blas
# BLAS_LIB := /path/to/your/blas

###############################################################################
# PYTHON Header path
PYTHON_HEADER_DIR := $(shell python -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())')
PYTORCH_INCLUDES := $(shell python -c 'from torch.utils.cpp_extension import include_paths; [print(p) for p in include_paths()]')
PYTORCH_LIBRARIES := $(shell python -c 'from torch.utils.cpp_extension import library_paths; [print(p) for p in library_paths()]')

# HEADER DIR is in pythonX.Xm folder
INCLUDE_DIRS := $(PYTHON_HEADER_DIR)
INCLUDE_DIRS += $(PYTHON_HEADER_DIR)/..
INCLUDE_DIRS += $(PYTORCH_INCLUDES)
LIBRARY_DIRS := $(PYTORCH_LIBRARIES)

# Determine ABI support
WITH_ABI := $(shell python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')

ifneq ($(CPU_ONLY), 1)
	# CUDA ROOT DIR that contains bin/ lib64/ and include/
	# CUDA_DIR := /usr/local/cuda
	CUDA_DIR := $(shell python -c 'from torch.utils.cpp_extension import _find_cuda_home; print(_find_cuda_home())')
	
	INCLUDE_DIRS += ./ $(CUDA_DIR)/include
	LIBRARY_DIRS += $(CUDA_DIR)/lib64
endif

SRC_DIR := ./src
OBJ_DIR := ./objs
CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/cuda/%.o,$(CU_SRCS))
STATIC_LIB := $(OBJ_DIR)/lib$(EXTENSION_NAME).a

# We will also explicitly add stdc++ to the link target.
LIBRARIES := stdc++ c10 caffe2 torch torch_python _C
ifneq ($(CPU_ONLY), 1)
	LIBRARIES += cudart cublas caffe2_gpu c10_cuda
	# Deprecate 3.0 for device memcpy
	CUDA_ARCH := -gencode arch=compute_35,code=sm_35 \
			-gencode=arch=compute_50,code=sm_50 \
			-gencode=arch=compute_52,code=sm_52 \
			-gencode=arch=compute_60,code=sm_60 \
			-gencode=arch=compute_61,code=sm_61 \
			-gencode=arch=compute_70,code=sm_70 \
			-gencode=arch=compute_75,code=sm_75 \
			-gencode=arch=compute_75,code=compute_75
endif

# BLAS configuration
BLAS ?= open
ifeq ($(BLAS), mkl)
	# MKL
	LIBRARIES += mkl_rt
	COMMON_FLAGS += -DUSE_MKL
	MKLROOT ?= /opt/intel/mkl
	BLAS_INCLUDE ?= $(MKLROOT)/include
	BLAS_LIB ?= $(MKLROOT)/lib $(MKLROOT)/lib/intel64
else ifeq ($(BLAS), open)
	# OpenBLAS
	LIBRARIES += openblas
else ifeq ($(BLAS), cblas)
	# OpenBLAS
	LIBRARIES += cblas
else
	# ATLAS
	LIBRARIES += atlas
	ATLAS_PATH := $(shell python -c "import numpy.distutils.system_info as si; ai = si.atlas_info(); [print(p) for p in ai.get_lib_dirs()]")
	LIBRARY_DIRS += $(ATLAS_PATH)
endif

# Debugging
ifeq ($(DEBUG), 1)
	COMMON_FLAGS += -DDEBUG -g -O0
	# https://gcoe-dresden.de/reaching-the-shore-with-a-fog-warning-my-eurohack-day-4-morning-session/
	NVCCFLAGS := -g -G # -rdc true
else
	COMMON_FLAGS += -DNDEBUG -O3
endif

WARNINGS := -Wall -Wcomment -Wno-sign-compare -Wno-deprecated-declarations

INCLUDE_DIRS += $(BLAS_INCLUDE)
LIBRARY_DIRS += $(BLAS_LIB)

# Automatic dependency generation (nvcc is handled separately)
CXXFLAGS += -MMD -MP

# Complete build flags.
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir)) \
	     -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=$(EXTENSION_NAME) \
	     -D_GLIBCXX_USE_CXX11_ABI=$(WITH_ABI)
CXXFLAGS += -fopenmp -fPIC -fwrapv -std=c++11 $(COMMON_FLAGS) $(WARNINGS)
NVCCFLAGS += -std=c++11 -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
LINKFLAGS += -pthread -fPIC $(WARNINGS) -Wl,-rpath=$(PYTHON_LIB_DIR) -Wl,--no-as-needed -Wl,--sysroot=/
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
	   $(foreach library,$(LIBRARIES),-l$(library))

ifeq ($(CPU_ONLY), 1)
	ALL_OBJS := $(OBJS)
	CXXFLAGS += -DCPU_ONLY
else
	ALL_OBJS := $(OBJS) $(CU_OBJS)
endif

all: $(STATIC_LIB)
	$(RM) -rf build dist

$(OBJ_DIR):
	@ mkdir -p $@
	@ mkdir -p $@/cuda

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@ echo CXX $<
	$(Q)$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJ_DIR)/cuda/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	@ echo NVCC $<
	$(Q)nvcc $(NVCCFLAGS) $(CUDA_ARCH) -M $< -o ${@:.o=.d} \
		-odir $(@D)
	$(Q)nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(STATIC_LIB): $(ALL_OBJS) | $(OBJ_DIR)
	$(RM) -f $(STATIC_LIB)
	@ echo LD -o $@
	ar rc $(STATIC_LIB) $(ALL_OBJS)

clean:
	@- $(RM) -rf $(OBJ_DIR) build dist
