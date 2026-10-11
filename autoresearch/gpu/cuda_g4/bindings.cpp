// PyTorch bridge: validate tensors, allocate output, dispatch native CUDA.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

extern "C" cudaError_t launch_swiglu(const void*, const void*, void*, int64_t, cudaStream_t);

torch::Tensor swiglu_cuda(torch::Tensor g, torch::Tensor u) {
    TORCH_CHECK(g.is_cuda() && u.is_cuda(), "CUDA required");
    TORCH_CHECK(g.scalar_type() == at::kBFloat16 && u.scalar_type() == at::kBFloat16, "BF16 required");
    TORCH_CHECK(g.device() == u.device() && g.sizes() == u.sizes(), "matching shape/device required");
    TORCH_CHECK(g.is_contiguous() && u.is_contiguous(), "contiguous required");
    const c10::cuda::CUDAGuard guard(g.device());
    auto out = torch::empty_like(g);
    C10_CUDA_CHECK(launch_swiglu(g.data_ptr(), u.data_ptr(), out.data_ptr(),
                               g.numel(), at::cuda::getCurrentCUDAStream()));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_cuda", &swiglu_cuda);
}
