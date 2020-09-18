#include "../helper_cuda.h"
#include "config.h"

#pragma once

namespace slab_hash {

class Allocator {
public:
    Allocator(int device_id = 0) : device_id_(device_id) {}
    template <typename T>
    T* allocate(size_t size) {}

    template <typename T>
    void deallocate(T* ptr) {}

protected:
    int device_id_;
};

class CudaAllocator: public Allocator {
public:
    CudaAllocator(int device_id = 0) : Allocator(device_id) {}
    template <typename T>
    T* allocate(size_t size) {
        T* ptr;
        CHECK_CUDA(cudaMalloc((void**)&ptr, sizeof(T) * size));
        return ptr;
    }

    template <typename T>
    void deallocate(T* ptr) {
        CHECK_CUDA(cudaFree(ptr));
    }
};

/**
class PyTorchAllocator: public Allocator {
public:
    PyTorchAllocator(int device_id = 0) : Allocator(device_id) {}

    template <typename T>
    T* allocate(size_t size)  {
        CHECK_CUDA(cudaGetDevice(&device_id_));
        auto options = torch::TensorOptions()
                               .dtype(torch::kInt8)
                               .device(torch::kCUDA, device_id_)
                               .requires_grad(false);
        tensor_ = torch::zeros(sizeof(T) * size, options);
        return tensor_.data<T>()
    }

    template <typename T>
    void deallocate(T* ptr)  {
        // let PyTorch handle this
    }

protected:
    torch::Tensor tensor_;
};
**/

} // namespace slab_hash
