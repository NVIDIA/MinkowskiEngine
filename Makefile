###############################################################################
# Uncomment for debugging
# DEBUG := 1
# Pretty build
Q ?= @

# Uncomment for CPU only build. From the command line, `python setup.py install --cpu_only`
# CPU_ONLY := 1

CXX ?= g++
PYTHON ?= python

EXTENSION_NAME := minkowski

# BLAS choice:
# atlas for ATLAS
# blas for default blas
# mkl for MKL. For conda, conda install -c intel mkl mkl-include
# openblas for OpenBlas (default)
BLAS ?= openblas
CUDA_HOME ?= $(shell $(PYTHON) -c 'from torch.utils.cpp_extension import _find_cuda_home; print(_find_cuda_home())')

# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
# Leave commented to accept the defaults for your choice of BLAS
# (which should work)!
# BLAS_INCLUDE_DIRS ?=
# BLAS_LIBRARY_DIRS ?=

###############################################################################
# PYTHON Header path
PYTHON_HEADER_DIR := $(shell $(PYTHON) -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())')
PYTORCH_INCLUDES := $(shell $(PYTHON) -c 'from torch.utils.cpp_extension import include_paths; [print(p) for p in include_paths()]')
PYTORCH_LIBRARIES := $(shell $(PYTHON) -c 'from torch.utils.cpp_extension import library_paths; [print(p) for p in library_paths()]')

# HEADER DIR is in pythonX.Xm folder
INCLUDE_DIRS := $(PYTHON_HEADER_DIR)
INCLUDE_DIRS += $(PYTHON_HEADER_DIR)/..
INCLUDE_DIRS += $(PYTORCH_INCLUDES)
LIBRARY_DIRS := $(PYTORCH_LIBRARIES)

# Determine ABI support
WITH_ABI := $(shell $(PYTHON) -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')

# Determine platform
UNAME := $(shell uname -s)
ifeq ($(UNAME), Linux)
	LINUX := 1
else ifeq ($(UNAME), Darwin)
	OSX := 1
	OSX_MAJOR_VERSION := $(shell sw_vers -productVersion | cut -f 1 -d .)
	OSX_MINOR_VERSION := $(shell sw_vers -productVersion | cut -f 2 -d .)

	CXX := /usr/local/opt/llvm/bin/clang
	# brew install llvm libomp
	INCLUDE_DIRS += /usr/local/opt/llvm/include
	LIBRARY_DIRS += /usr/local/opt/llvm/lib
endif

ifneq ($(CPU_ONLY), 1)
	# CUDA ROOT DIR that contains bin/ lib64/ and include/
	# CUDA_HOME := /usr/local/cuda
	
	NVCC ?= $(CUDA_HOME)/bin/nvcc
	INCLUDE_DIRS += ./ $(CUDA_HOME)/include
	LIBRARY_DIRS += $(CUDA_HOME)/lib64
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
	LIBRARIES += cudart cublas cusparse caffe2_gpu c10_cuda
	CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
			-gencode arch=compute_35,code=sm_35 \
			-gencode=arch=compute_50,code=sm_50 \
			-gencode=arch=compute_52,code=sm_52 \
			-gencode=arch=compute_60,code=sm_60 \
			-gencode=arch=compute_61,code=sm_61 \
			-gencode=arch=compute_70,code=sm_70 \
			-gencode=arch=compute_75,code=sm_75 \
			-gencode=arch=compute_75,code=compute_75
endif

# BLAS configuration: mkl, atlas, open, blas
BLAS ?= openblas
ifeq ($(BLAS), mkl)
	# MKL
	LIBRARIES += mkl_rt
	COMMON_FLAGS += -DUSE_MKL
	MKLROOT ?= /opt/intel/mkl
	BLAS_INCLUDE_DIRS ?= $(MKLROOT)/include
	BLAS_LIBRARY_DIRS ?= $(MKLROOT)/lib $(MKLROOT)/lib/intel64
else ifeq ($(BLAS), openblas)
	# OpenBLAS
	LIBRARIES += openblas
else ifeq ($(BLAS), blas)
	# OpenBLAS
	LIBRARIES += blas
else
	# ATLAS
	LIBRARIES += atlas
	ATLAS_PATH := $(shell $(PYTHON) -c "import numpy.distutils.system_info as si; ai = si.atlas_info(); [print(p) for p in ai.get_lib_dirs()]")
	BLAS_LIBRARY_DIRS += $(ATLAS_PATH)
endif

INCLUDE_DIRS += ./src/3rdparty
INCLUDE_DIRS += $(BLAS_INCLUDE_DIRS)
LIBRARY_DIRS += $(BLAS_LIBRARY_DIRS)

# Debugging
ifeq ($(DEBUG), 1)
	COMMON_FLAGS += -DDEBUG -g -O0
	# https://gcoe-dresden.de/reaching-the-shore-with-a-fog-warning-my-eurohack-day-4-morning-session/
	NVCCFLAGS := -g -G # -rdc true
else
	COMMON_FLAGS += -DNDEBUG -O3
endif

WARNINGS := -Wall -Wcomment -Wno-sign-compare -Wno-deprecated-declarations

# Automatic dependency generation (nvcc is handled separately)
CXXFLAGS += -MMD -MP

# Fast math
CXXFLAGS += -ffast-math -funsafe-math-optimizations -fno-math-errno

# BATCH FIRST
CXXFLAGS += -DBATCH_FIRST=1

# Complete build flags.
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir)) \
	     -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=$(EXTENSION_NAME) \
	     -D_GLIBCXX_USE_CXX11_ABI=$(WITH_ABI)

CXXFLAGS += -fopenmp -fPIC -fwrapv -std=c++14 $(COMMON_FLAGS) $(WARNINGS)
NVCCFLAGS += -std=c++14 -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
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
	$(Q)$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -M $< -o ${@:.o=.d} \
		-odir $(@D)
	$(Q)$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(STATIC_LIB): $(ALL_OBJS) | $(OBJ_DIR)
	$(RM) -f $(STATIC_LIB)
	@ echo LD -o $@
	ar rc $(STATIC_LIB) $(ALL_OBJS)

clean:
	@- $(RM) -rf $(OBJ_DIR) build dist
