// Native CUDA SwiGLU for A100. No PyTorch/ATen arithmetic.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

__global__ void swiglu_kernel(const __nv_bfloat16* g, const __nv_bfloat16* u,
                              __nv_bfloat16* out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = __bfloat162float(g[i]);
    // Match F.silu(g) * u: round SiLU to BF16 before multiplication.
    float activated = __bfloat162float(__float2bfloat16_rn(x / (1.0f + expf(-x))));
    out[i] = __float2bfloat16_rn(activated * __bfloat162float(u[i]));
}

extern "C" cudaError_t launch_swiglu(const void* g, const void* u, void* out,
                                    int64_t n, cudaStream_t stream) {
    if (n == 0) return cudaSuccess;
    swiglu_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(g),
        static_cast<const __nv_bfloat16*>(u),
        static_cast<__nv_bfloat16*>(out), n);
    return cudaGetLastError();
}
