#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

// ---------------------------------------------------------------------------
// CUDA Kernel
//   Parallel over B and D
//   Processes L sequentially for each (b, d)
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void
ssm_scan_fwd_kernel(const scalar_t *__restrict__ x,        // (B, L, D)
                    const scalar_t *__restrict__ dt,       // (B, L, D)
                    const scalar_t *__restrict__ A,        // (D, N)
                    const scalar_t *__restrict__ B_params, // (B, L, N)
                    const scalar_t *__restrict__ C_params, // (B, L, N)
                    const scalar_t *__restrict__ D_params, // (D)
                    scalar_t *__restrict__ y,              // (B, L, D)
                    int B, int L, int D, int N) {

  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int d = blockIdx.y * blockDim.y + threadIdx.y;

  if (b < B && d < D) {
    // Strides
    int stride_bL = L * D;
    int stride_bLN = L * N;

    // Shared or local hidden state buffer for this thread
    // N is typically small (e.g., 16), so we use a local array
    float h[64]; // Support N up to 64
    for (int n = 0; n < N; ++n)
      h[n] = 0.0f;

    const scalar_t D_val = D_params[d];
    const scalar_t *A_d = A + d * N;

    for (int t = 0; t < L; ++t) {
      int idx_bt = b * stride_bL + t * D + d;
      int idx_btn_base = b * stride_bLN + t * N;

      scalar_t x_val = x[idx_bt];
      scalar_t dt_val = dt[idx_bt];

      float y_val_local = 0.0f;
      for (int n = 0; n < N; ++n) {
        float a_val = (float)A_d[n];
        float b_val = (float)B_params[idx_btn_base + n];
        float c_val = (float)C_params[idx_btn_base + n];

        float a_bar = expf(dt_val * a_val);
        float b_bar = dt_val * b_val;

        h[n] = a_bar * h[n] + b_bar * (float)x_val;
        y_val_local += h[n] * c_val;
      }

      y[idx_bt] = (scalar_t)y_val_local + D_val * x_val;
    }
  }
}

// ---------------------------------------------------------------------------
// C++ Wrapper for CUDA Kernel
// ---------------------------------------------------------------------------
torch::Tensor ssm_scan_fwd_cuda(torch::Tensor x, torch::Tensor dt,
                                torch::Tensor A, torch::Tensor B_params,
                                torch::Tensor C_params,
                                torch::Tensor D_params) {

  const int B = x.size(0);
  const int L = x.size(1);
  const int D = x.size(2);
  const int N = A.size(1);

  auto y = torch::empty_like(x);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((B + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (D + threadsPerBlock.y - 1) / threadsPerBlock.y);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "ssm_scan_fwd_cuda", ([&] {
        ssm_scan_fwd_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            x.data_ptr<scalar_t>(), dt.data_ptr<scalar_t>(),
            A.data_ptr<scalar_t>(), B_params.data_ptr<scalar_t>(),
            C_params.data_ptr<scalar_t>(), D_params.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(), B, L, D, N);
      }));

  return y;
}
