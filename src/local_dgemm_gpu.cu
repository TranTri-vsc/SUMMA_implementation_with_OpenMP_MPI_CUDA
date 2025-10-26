#include "localmatrix.hpp"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>

#ifdef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

// C(MxN) += A(MxK) * B(KxN), row-major
__global__ void dgemm_accum_kernel(const double* __restrict__ A,
                                   const double* __restrict__ B,
                                   double* __restrict__ C,
                                   int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    double acc = 0.0;
    int a_base = row * K;
    for (int k = 0; k < K; ++k) {
        acc += A[a_base + k] * B[k * N + col];
    }
    // Accumulate into C (unique element per thread; no atomics needed)
    C[row * N + col] += acc;
}

// Write your local DGEMM using CUDA
extern "C"
void local_dgemm_gpu(double* dC, int Cm, int Cn,
                     const double* dA, int Am, int An,
                     const double* dB, int Bm, int Bn,
                     cudaStream_t stream/*=0*/)
{
    // Expect: A(Am x An), B(Bm x Bn), C(Cm x Cn), with Am==Cm, Bn==Cn, An==Bm
    if (!(Am == Cm && Bn == Cn && An == Bm)) {
        // Silent return on mismatch; upstream code can guard if desired.
        return;
    }

    dim3 block(16, 16, 1);
    dim3 grid((Cn + block.x - 1) / block.x,
              (Cm + block.y - 1) / block.y,
              1);

    // K = An (== Bm)
    dgemm_accum_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, Cm, Cn, An);
}

#endif // ENABLE_CUDA
