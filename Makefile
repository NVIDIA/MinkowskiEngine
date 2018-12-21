###############################################################################
# Uncomment for debugging
# DEBUG := 1
# Pretty build
# Q ?= @

CXX := g++

# CUDA ROOT DIR that contains bin/ lib64/ and include/
CUDA_DIR := /usr/local/cuda

INCLUDE_DIRS := ./ $(CUDA_DIR)/include
LIBRARY_DIRS := $(CUDA_DIR)/lib64

PYTHON_HEADER_PATH := /home/chrischoy/anaconda3/envs/py3-mink-dev/include/python3.7m
PYTHON_LIB_PATH := /home/chrischoy/anaconda3/envs/py3-mink-dev/lib
PYTHON_PACKAGE_PATH := /home/chrischoy/anaconda3/envs/py3-mink-dev/lib/python3.7/site-packages

INCLUDE_DIRS += $(PYTHON_PACKAGE_PATH)/torch/lib/include/
INCLUDE_DIRS += $(PYTHON_PACKAGE_PATH)/torch/lib/include/TH
INCLUDE_DIRS += $(PYTHON_PACKAGE_PATH)/torch/lib/include/THC
INCLUDE_DIRS += $(PYTHON_PACKAGE_PATH)/torch/lib/include/torch/csrc/api/include
INCLUDE_DIRS += $(PYTHON_HEADER_PATH)

LIBRARY_DIRS += $(PYTHON_LIB_PATH)
LIBRARY_DIRS += $(PYTHON_PACKAGE_PATH)/torch/lib/
LIBRARY_DIRS += $(PYTHON_PACKAGE_PATH)/torch/

# BLAS choice:
# atlas for ATLAS
# mkl for MKL
# open for OpenBlas (default)
BLAS := open

# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
# Leave commented to accept the defaults for your choice of BLAS
# (which should work)!
# BLAS_INCLUDE := /path/to/your/blas
# BLAS_LIB := /path/to/your/blas

###############################################################################
SRC_DIR := ./src
OBJ_DIR := ./MinkowskiEngineObjs
CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/cuda/%.o,$(CU_SRCS))
STATIC_LIB := $(OBJ_DIR)/libminkowski.so

# CUDA architecture setting: going with all of them.
# For CUDA < 6.0, comment the *_50 through *_61 lines for compatibility.
# For CUDA < 8.0, comment the *_60 and *_61 lines for compatibility.
CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61

# We will also explicitly add stdc++ to the link target.
LIBRARIES += stdc++ cudart cublas c10 caffe2 torch torch_python caffe2_gpu

# BLAS configuration (default = open)
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
else
	# ATLAS
	# Linux simply has cblas and atlas
	LIBRARIES += cblas atlas
endif

# Debugging
ifeq ($(DEBUG), 1)
	COMMON_FLAGS += -DDEBUG -g -O0
	# https://gcoe-dresden.de/reaching-the-shore-with-a-fog-warning-my-eurohack-day-4-morning-session/
	NVCCFLAGS += -G -rdc true
else
	COMMON_FLAGS += -DNDEBUG -O2
endif

WARNINGS := -Wall -Wno-sign-compare -Wstrict-prototypes

INCLUDE_DIRS += $(BLAS_INCLUDE)
LIBRARY_DIRS += $(BLAS_LIB)

# Automatic dependency generation (nvcc is handled separately)
CXXFLAGS += -MMD -MP

# Complete build flags.
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -pthread -B /home/chrischoy/anaconda3/envs/py3-mink-dev/compiler_compat -fPIC $(COMMON_FLAGS) $(WARNINGS) -fwrapv -O3 -D_GLIBCXX_USE_CXX11_ABI=0 -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=minkowski -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
NVCCFLAGS += -std=c++11 -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS) -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=minkowski -D_GLIBCXX_USE_CXX11_ABI=0

LINKFLAGS += -pthread -B /home/chrischoy/anaconda3/envs/py3-mink-dev/compiler_compat -fPIC $(WARNINGS) -Wl,-rpath=/home/chrischoy/anaconda3/envs/py3-mink-dev/lib -Wl,--no-as-needed -Wl,--sysroot=/ -D_GLIBCXX_USE_CXX11_ABI=0
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
	   $(foreach library,$(LIBRARIES),-l$(library)) -l:_C.cpython-37m-x86_64-linux-gnu.so

all: $(STATIC_LIB)
	python setup.py install --force

$(OBJ_DIR):
	@ mkdir -p $@
	@ mkdir -p $@/cuda

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@ echo CXX $<
	$(Q)nvcc $(NVCCFLAGS) $(CUDA_ARCH) -M $< -o ${@:.o=.d} \
		-odir $(@D)
	$(Q)nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(OBJ_DIR)/cuda/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	@ echo NVCC $<
	$(Q)nvcc $(NVCCFLAGS) $(CUDA_ARCH) -M $< -o ${@:.o=.d} \
		-odir $(@D)
	$(Q)nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(STATIC_LIB): $(OBJS) $(CU_OBJS) | $(OBJ_DIR)
	$(RM) -f $(STATIC_LIB)
	$(RM) -rf build dist
	@ echo LD -o $@
	ar rc $(STATIC_LIB) $(OBJS) $(CU_OBJS)
	# $(Q)$(CXX) -o $@ $(OBJS) $(CU_OBJS) $(LINKFLAGS) $(LDFLAGS)

clean:
	@- $(RM) -rf $(OBJ_DIR) build dist
