#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_bf16.h>
using at::Tensor;
using bf = __nv_bfloat16;
static const bf *rd(const Tensor &x) {
  return reinterpret_cast<const bf *>(x.data_ptr<at::BFloat16>());
}
static bf *wr(const Tensor &x) {
  return reinterpret_cast<bf *>(x.data_ptr<at::BFloat16>());
}
static cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }
static void check(const Tensor &x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 &&
                  x.is_contiguous(),
              "expected contiguous CUDA BF16 tensor");
}
static void blas(cublasStatus_t status) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLAS status ", int(status));
}
static cublasHandle_t handle() {
  auto h = at::cuda::getCurrentCUDABlasHandle();
  blas(cublasSetStream(h, stream()));
  blas(cublasSetMathMode(h, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
  return h;
}
int cublas_version() {
  int v;
  blas(cublasGetVersion(handle(), &v));
  return v;
}
__device__ float sum_warp(float x) {
  for (int d = 16; d; d /= 2)
    x += __shfl_down_sync(0xffffffff, x, d);
  return x;
}
__device__ float sum_block(float x) {
  __shared__ float s[8];
  int l = threadIdx.x % 32, w = threadIdx.x / 32;
  x = sum_warp(x);
  if (l == 0)
    s[w] = x;
  __syncthreads();
  x = threadIdx.x < 8 ? s[l] : 0;
  if (w == 0)
    x = sum_warp(x);
  if (threadIdx.x == 0)
    s[0] = x;
  __syncthreads();
  float result = s[0];
  __syncthreads();
  return result;
}
__device__ float max_block(float x) {
  __shared__ float s[8];
  int l = threadIdx.x % 32, w = threadIdx.x / 32;
  for (int d = 16; d; d /= 2)
    x = fmaxf(x, __shfl_down_sync(0xffffffff, x, d));
  if (l == 0)
    s[w] = x;
  __syncthreads();
  x = threadIdx.x < 8 ? s[l] : -INFINITY;
  if (w == 0)
    for (int d = 16; d; d /= 2)
      x = fmaxf(x, __shfl_down_sync(0xffffffff, x, d));
  if (threadIdx.x == 0)
    s[0] = x;
  __syncthreads();
  float result = s[0];
  __syncthreads();
  return result;
}
__global__ void lb_embedding(const int64_t *tok, const bf *table, bf *out,
                             int T, int D, int64_t sb, int64_t st) {
  int r = blockIdx.x;
  auto id = tok[(r / T) * sb + (r % T) * st];
  for (int d = threadIdx.x; d < D; d += blockDim.x)
    out[int64_t(r) * D + d] = table[id * D + d];
}
void embedding(Tensor t, Tensor w, Tensor o) {
  check(w);
  c10::cuda::CUDAGuard guard(w.device());
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kLong && t.dim() == 2,
              "expected CUDA int64 tokens[B,T]");
  lb_embedding<<<t.numel(), 256, 0, stream()>>>(t.data_ptr<int64_t>(), rd(w),
                                                wr(o), t.size(1), w.size(1),
                                                t.stride(0), t.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void lb_norm(const bf *x, const bf *r, const bf *w, const bf *b,
                        bf *summed, bf *out, int D, float eps, bool add) {
  int64_t base = int64_t(blockIdx.x) * D;
  float total = 0;
  for (int d = threadIdx.x; d < D; d += 256) {
    float v = __bfloat162float(x[base + d]);
    if (add) {
      v = __bfloat162float(
          __float2bfloat16_rn(v + __bfloat162float(r[base + d])));
      summed[base + d] = __float2bfloat16_rn(v);
    }
    total += v;
  }
  float mean = sum_block(total) / D;
  float var = 0;
  for (int d = threadIdx.x; d < D; d += 256) {
    float v = __bfloat162float(add ? summed[base + d] : x[base + d]) - mean;
    var += v * v;
  }
  float inv = rsqrtf(sum_block(var) / D + eps);
  for (int d = threadIdx.x; d < D; d += 256) {
    float v = __bfloat162float(add ? summed[base + d] : x[base + d]);
    out[base + d] = __float2bfloat16_rn(
        (v - mean) * inv * __bfloat162float(w[d]) + __bfloat162float(b[d]));
  }
}
void norm(Tensor x, Tensor r, Tensor w, Tensor b, Tensor s, Tensor o,
          double eps, bool add) {
  check(x);
  check(r);
  check(w);
  check(b);
  c10::cuda::CUDAGuard guard(x.device());
  lb_norm<<<x.numel() / x.size(-1), 256, 0, stream()>>>(
      rd(x), rd(r), rd(w), rd(b), wr(s), wr(o), x.size(-1), eps, add);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void lb_swiglu(const bf *x, bf *o, int64_t n, int F) {
  int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
  if (i >= n)
    return;
  int64_t j = (i / F) * 2 * F + i % F;
  float g = __bfloat162float(x[j]);
  o[i] = __float2bfloat16_rn(g / (1 + expf(-g)) * __bfloat162float(x[j + F]));
}
void swiglu(Tensor x, Tensor o) {
  check(x);
  c10::cuda::CUDAGuard guard(x.device());
  lb_swiglu<<<(o.numel() + 255) / 256, 256, 0, stream()>>>(
      rd(x), wr(o), o.numel(), o.size(-1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void lb_rope(const bf *x, const float *c, const float *s,
                        const int64_t *pos, bf *q, bf *k, bf *v, int T, int N,
                        int K, int H, int S, bool cache) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= T * H)
    return;
  int t = i / H, d = i % H, head = blockIdx.y % (N + K),
      b = blockIdx.y / (N + K), p = pos ? int(*pos) : 0;
  int half = H / 2, width = (N + 2 * K) * H;
  int64_t offset = int64_t(b * T + t) * width + head * H;
  float a = __bfloat162float(x[offset + d]),
        z = __bfloat162float(x[offset + (d < half ? d + half : d - half)]);
  float co = c[(p + t) * half + d % half], si = s[(p + t) * half + d % half];
  bf rot = __float2bfloat16_rn(a * co + (d < half ? -z : z) * si);
  if (head < N)
    q[((int64_t(b) * N + head) * T + t) * H + d] = rot;
  else {
    int dst = cache ? p + t : t;
    if (dst < 0 || dst >= S)
      return;
    int64_t j = ((int64_t(b) * K + head - N) * S + dst) * H + d;
    k[j] = rot;
    v[j] = x[int64_t(b * T + t) * width + (head + K) * H + d];
  }
}
void rope(Tensor x, Tensor c, Tensor s, c10::optional<Tensor> p, Tensor q,
          Tensor k, Tensor v, bool cache) {
  check(x);
  check(k);
  check(v);
  c10::cuda::CUDAGuard guard(x.device());
  int H = q.size(3), T = x.size(1), N = q.size(1), K = k.size(1);
  TORCH_CHECK(H % 2 == 0, "even head dim required");
  lb_rope<<<dim3((T * H + 255) / 256, x.size(0) * (N + K)), 256, 0, stream()>>>(
      rd(x), c.data_ptr<float>(), s.data_ptr<float>(),
      p ? p->data_ptr<int64_t>() : nullptr, wr(q), wr(k), wr(v), T, N, K, H,
      k.size(2), cache);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void lb_argmax(const bf *x, int64_t *out, int T, int V, int64_t sb,
                          int64_t st, int64_t sv) {
  __shared__ float values[256];
  __shared__ int ids[256];
  int tid = threadIdx.x, b = blockIdx.x, idx = INT_MAX;
  float best = -INFINITY;
  for (int i = tid; i < V; i += 256) {
    float a = __bfloat162float(x[b * sb + (T - 1) * st + i * sv]);
    if (a > best || (a == best && i < idx)) {
      best = a;
      idx = i;
    }
  }
  values[tid] = best;
  ids[tid] = idx;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (tid < d) {
      float a = values[tid + d];
      int j = ids[tid + d];
      if (a > values[tid] || (a == values[tid] && j < ids[tid])) {
        values[tid] = a;
        ids[tid] = j;
      }
    }
    __syncthreads();
  }
  if (tid == 0)
    out[b] = ids[0];
}
void argmax(Tensor x, Tensor o) {
  c10::cuda::CUDAGuard guard(x.device());
  lb_argmax<<<x.size(0), 256, 0, stream()>>>(rd(x), o.data_ptr<int64_t>(),
                                             x.size(1), x.size(2), x.stride(0),
                                             x.stride(1), x.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void lb_advance(int64_t *p) { ++*p; }
void advance(Tensor p) {
  c10::cuda::CUDAGuard guard(p.device());
  lb_advance<<<1, 1, 0, stream()>>>(p.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void lb_copy_prefix(const bf *src, bf *dst, int64_t n, int T, int S,
                               int H) {
  int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
  if (i < n)
    dst[(i / (T * H)) * S * H + i % (T * H)] = src[i];
}
void copy_prefix(Tensor s, Tensor d) {
  check(s);
  check(d);
  TORCH_CHECK(s.size(2) <= d.size(2), "prefix exceeds capacity");
  c10::cuda::CUDAGuard guard(s.device());
  lb_copy_prefix<<<(s.numel() + 255) / 256, 256, 0, stream()>>>(
      rd(s), wr(d), s.numel(), s.size(2), d.size(2), s.size(3));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void lb_copy_token(const int64_t *s, int64_t *d, int n) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i < n)
    d[i] = s[i];
}
void copy_token(Tensor s, Tensor d) {
  TORCH_CHECK(s.is_contiguous() && d.is_contiguous() && s.numel() == d.numel(),
              "token copy shape");
  c10::cuda::CUDAGuard guard(s.device());
  lb_copy_token<<<(s.numel() + 255) / 256, 256, 0, stream()>>>(
      s.data_ptr<int64_t>(), d.data_ptr<int64_t>(), s.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void linear(Tensor x, Tensor w, Tensor o) {
  check(x);
  check(w);
  TORCH_CHECK(x.size(-1) == w.size(0), "linear inner dimension");
  c10::cuda::CUDAGuard guard(x.device());
  int K = w.size(0), N = w.size(1), M = x.numel() / K;
  float a = 1, b = 0;
  blas(cublasGemmEx(handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &a, rd(w),
                    CUDA_R_16BF, N, rd(x), CUDA_R_16BF, K, &b, wr(o),
                    CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
__global__ void lb_softmax(const float *scores, bf *p, int T, int S, int N) {
  int row = blockIdx.x, qt = row % T;
  float m = -INFINITY;
  for (int j = threadIdx.x; j <= qt && j < S; j += 256)
    m = fmaxf(m, scores[int64_t(row) * S + j]);
  m = max_block(m);
  float z = 0;
  for (int j = threadIdx.x; j <= qt && j < S; j += 256)
    z += expf(scores[int64_t(row) * S + j] - m);
  z = sum_block(z);
  for (int j = threadIdx.x; j < S; j += 256)
    p[int64_t(row) * S + j] = __float2bfloat16_rn(
        j <= qt ? expf(scores[int64_t(row) * S + j] - m) / z : 0);
}
__global__ void lb_heads(const bf *x, bf *o, int64_t total, int N, int T,
                         int H) {
  int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
  if (i >= total)
    return;
  int d = i % H, h = (i / H) % N, t = (i / (H * N)) % T, b = i / (H * N * T);
  o[i] = x[((int64_t(b) * N + h) * T + t) * H + d];
}
void prefill_attention(Tensor q, Tensor k, Tensor v, Tensor scores,
                       Tensor probs, Tensor heads, Tensor out) {
  check(q);
  check(k);
  check(v);
  c10::cuda::CUDAGuard guard(q.device());
  int B = q.size(0), N = q.size(1), K = k.size(1), T = q.size(2), S = k.size(2),
      H = q.size(3);
  TORCH_CHECK(N % K == 0 && S == T, "prefill attention dimensions");
  auto h = handle();
  float a = 1 / std::sqrt(float(H)), zero = 0, one = 1;
  // Grouped-query heads share K/V. Each group is a strided batch across KV
  // heads.
  for (int b = 0; b < B; b++)
    for (int g = 0; g < N / K; g++) {
      blas(cublasGemmStridedBatchedEx(
          h, CUBLAS_OP_T, CUBLAS_OP_N, S, T, H, &a,
          rd(k) + int64_t(b) * K * S * H, CUDA_R_16BF, H, int64_t(S) * H,
          rd(q) + (int64_t(b) * N + g) * T * H, CUDA_R_16BF, H,
          int64_t(N / K) * T * H, &zero,
          scores.data_ptr<float>() + (int64_t(b) * N + g) * T * S, CUDA_R_32F,
          S, int64_t(N / K) * T * S, K, CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT));
    }
  lb_softmax<<<B * N * T, 256, 0, stream()>>>(scores.data_ptr<float>(),
                                              wr(probs), T, S, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  for (int b = 0; b < B; b++)
    for (int g = 0; g < N / K; g++) {
      blas(cublasGemmStridedBatchedEx(
          h, CUBLAS_OP_N, CUBLAS_OP_N, H, T, S, &one,
          rd(v) + int64_t(b) * K * S * H, CUDA_R_16BF, H, int64_t(S) * H,
          rd(probs) + (int64_t(b) * N + g) * T * S, CUDA_R_16BF, S,
          int64_t(N / K) * T * S, &zero,
          wr(heads) + (int64_t(b) * N + g) * T * H, CUDA_R_16BF, H,
          int64_t(N / K) * T * H, K, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
  lb_heads<<<(out.numel() + 255) / 256, 256, 0, stream()>>>(
      rd(heads), wr(out), out.numel(), N, T, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void lb_decode_parts(const bf *q, const bf *k, const bf *v,
                                const int64_t *pos, float *tmp, int N, int K,
                                int S, int H, int P) {
  int lane = threadIdx.x, head = blockIdx.y % N, b = blockIdx.y / N,
      part = blockIdx.x, kh = head / (N / K);
  int valid = min(S, int(*pos) + 1), chunk = (S + P - 1) / P, lo = part * chunk,
      hi = min(valid, lo + chunk);
  float acc[8] = {0}, qv[8];
  for (int d = 0; d < 8; d++)
    qv[d] =
        lane + d * 32 < H
            ? __bfloat162float(q[(int64_t(b) * N + head) * H + lane + d * 32])
            : 0;
  float m = -INFINITY, z = 0;
  for (int t = lo; t < hi; t++) {
    float dot = 0;
    for (int d = 0; d < 8; d++)
      if (lane + d * 32 < H)
        dot +=
            qv[d] * __bfloat162float(
                        k[((int64_t(b) * K + kh) * S + t) * H + lane + d * 32]);
    dot = sum_warp(dot);
    dot = __shfl_sync(0xffffffff, dot, 0) * rsqrtf(float(H));
    float next = fmaxf(m, dot), a = expf(m - next), p = expf(dot - next);
    z = z * a + p;
    for (int d = 0; d < 8; d++)
      if (lane + d * 32 < H)
        acc[d] =
            acc[d] * a +
            p * __bfloat162float(
                    v[((int64_t(b) * K + kh) * S + t) * H + lane + d * 32]);
    m = next;
  }
  int64_t base = ((int64_t(b) * N + head) * P + part) * (H + 2);
  if (lane == 0) {
    tmp[base + H] = m;
    tmp[base + H + 1] = z;
  }
  for (int d = 0; d < 8; d++)
    if (lane + d * 32 < H)
      tmp[base + lane + d * 32] = acc[d];
}
__global__ void lb_decode_reduce(const float *tmp, bf *out, int H, int P) {
  int bh = blockIdx.x, d = threadIdx.x;
  float m = -INFINITY;
  for (int p = 0; p < P; p++)
    m = fmaxf(m, tmp[(int64_t(bh) * P + p) * (H + 2) + H]);
  float z = 0, a = 0;
  for (int p = 0; p < P; p++) {
    int64_t off = (int64_t(bh) * P + p) * (H + 2);
    float f = expf(tmp[off + H] - m);
    z += tmp[off + H + 1] * f;
    if (d < H)
      a += tmp[off + d] * f;
  }
  if (d < H)
    out[int64_t(bh) * H + d] = __float2bfloat16_rn(a / z);
}
void decode_attention(Tensor q, Tensor k, Tensor v, Tensor pos, Tensor tmp,
                      Tensor out, int64_t parts) {
  check(q);
  check(k);
  check(v);
  c10::cuda::CUDAGuard guard(q.device());
  int N = q.size(1), K = k.size(1), H = q.size(3), S = k.size(2);
  TORCH_CHECK(H <= 256 && q.size(2) == 1 && N % K == 0,
              "decode requires T=1, H<=256 and grouped heads");
  lb_decode_parts<<<dim3(parts, q.size(0) * N), 32, 0, stream()>>>(
      rd(q), rd(k), rd(v), pos.data_ptr<int64_t>(), tmp.data_ptr<float>(), N, K,
      S, H, parts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  lb_decode_reduce<<<q.size(0) * N, 256, 0, stream()>>>(tmp.data_ptr<float>(),
                                                        wr(out), H, parts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void lb_record_token(const int64_t *t, int64_t *h, int B, int W,
                                int I) {
  int b = blockIdx.x * 256 + threadIdx.x;
  if (b < B)
    h[int64_t(b) * W + I] = t[b];
}
void record_token(Tensor t, Tensor h, int64_t i) {
  c10::cuda::CUDAGuard guard(t.device());
  lb_record_token<<<(t.numel() + 255) / 256, 256, 0, stream()>>>(
      t.data_ptr<int64_t>(), h.data_ptr<int64_t>(), t.numel(), h.size(1), i);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
