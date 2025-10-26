#include "summa.hpp"
#include "localmatrix.hpp"
#include "verify.hpp"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

extern "C"
void local_dgemm_gpu(double* dC,int Cm,int Cn,
                     const double* dA,int Am,int An,
                     const double* dB,int Bm,int Bn,
                     cudaStream_t stream);

// GPU SUMMA: host-staged MPI broadcasts, device compute with CUDA DGEMM.
// Accumulates C = A * B where A,B,C are 2D-block distributed across a P x P grid.
void run_summa_gpu(int N,const Dist2D& d, bool do_verify){
  int rank, world;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  // Allocate and initialize local matrices
  LocalMatrix A(N, N, d);
  LocalMatrix B(N, N, d);
  LocalMatrix C(N, N, d);

  A.initialize_A();
  B.initialize_B();
  C.zero();

  // Setup per-dimension partition info
  std::vector<int> row_sizes, row_offsets;
  std::vector<int> col_sizes, col_offsets;

  split_sizes(N, d.P, row_sizes, row_offsets);
  split_sizes(N, d.P, col_sizes, col_offsets);

  // Local dimensions for this rank
  const int local_rows = A.l_rows;   // rows of A and C
  const int local_cols = B.l_cols;   // cols of B and C

  // Allocate persistent device buffer for C
  double* dC = nullptr;
  const size_t C_size = static_cast<size_t>(local_rows) * local_cols * sizeof(double);
  cudaMalloc(&dC, C_size);
  cudaMemset(dC, 0, C_size);

  // Single CUDA stream for async ops
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // -------------------- TIMING SETUP --------------------
  // End-to-end timing (includes MPI + H2D + kernels)
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();

  // Compute-only timing on GPU via CUDA events (kernels and any work enqueued in 'stream' inside local_dgemm_gpu)
  float compute_ms_accum = 0.0f;
  cudaEvent_t comp_start, comp_stop;
  cudaEventCreate(&comp_start);
  cudaEventCreate(&comp_stop);
  // ------------------------------------------------------

  // SUMMA main loop over block index block_k
  for (int block_k = 0; block_k < d.P; block_k++) {
      const int shared_dim = col_sizes[block_k]; // width of A-panel, height of B-panel

      // 1. Prepare A panel (m × shared_dim), broadcast along row communicator
      const double* A_panel_host = nullptr;
      std::vector<double> A_buffer;

      if (d.myc == block_k) {
          A_panel_host = A.data.data();
      } else {
          A_buffer.resize(static_cast<size_t>(local_rows) * shared_dim);
          A_panel_host = A_buffer.data();
      }

      MPI_Bcast(const_cast<double*>(A_panel_host),
                local_rows * shared_dim, MPI_DOUBLE,
                block_k, d.row_comm);

      // 2. Prepare B panel (shared_dim × n), broadcast along column communicator
      const double* B_panel_host = nullptr;
      std::vector<double> B_buffer;

      if (d.myr == block_k) {
          B_panel_host = B.data.data();
      } else {
          B_buffer.resize(static_cast<size_t>(shared_dim) * local_cols);
          B_panel_host = B_buffer.data();
      }

      MPI_Bcast(const_cast<double*>(B_panel_host),
                shared_dim * local_cols, MPI_DOUBLE,
                block_k, d.col_comm);

      // 3. Copy both panels to device memory
      double *dA = nullptr, *dB = nullptr;
      const size_t bytesA = static_cast<size_t>(local_rows) * shared_dim * sizeof(double);
      const size_t bytesB = static_cast<size_t>(shared_dim) * local_cols * sizeof(double);

      cudaMalloc(&dA, bytesA);
      cudaMalloc(&dB, bytesB);

      cudaMemcpyAsync(dA, A_panel_host, bytesA, cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(dB, B_panel_host, bytesB, cudaMemcpyHostToDevice, stream);

      // 4. Local GPU computation: dC += dA * dB  (compute-only timing)
      cudaEventRecord(comp_start, stream);
      local_dgemm_gpu(
          dC, local_rows, local_cols,
          dA, local_rows, shared_dim,
          dB, shared_dim, local_cols,
          stream
      );
      cudaEventRecord(comp_stop, stream);
      cudaEventSynchronize(comp_stop);
      float iter_ms = 0.0f;
      cudaEventElapsedTime(&iter_ms, comp_start, comp_stop);
      compute_ms_accum += iter_ms;

      // 5. Free temporary device buffers
      cudaFree(dA);
      cudaFree(dB);
  }

  // End-to-end stop (make sure all device work is done before we read time)
  cudaStreamSynchronize(stream);
  const double t1 = MPI_Wtime();

  // -------------------- GFLOP/s REPORT --------------------
  // Reduce times to the slowest rank (the one that sets total runtime)
  const double loop_time_sec_local    = t1 - t0;
  const double compute_time_sec_local = static_cast<double>(compute_ms_accum) / 1000.0;

  double loop_time_sec_max    = 0.0;
  double compute_time_sec_max = 0.0;

  MPI_Reduce((void*)&loop_time_sec_local,    (void*)&loop_time_sec_max,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce((void*)&compute_time_sec_local, (void*)&compute_time_sec_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  const double global_flops = 2.0 * static_cast<double>(N) * N * N; // total work for full C = A*B
  if (rank == 0) {
      const double gflops_end_to_end = (global_flops / loop_time_sec_max)    / 1e9;
      const double gflops_compute    = (global_flops / compute_time_sec_max) / 1e9;

      std::cout << "[GPU] N=" << N << " P=" << d.P << " ranks=" << world << "\n"
                << "  End-to-end  : " << loop_time_sec_max    << " s, "
                << gflops_end_to_end << " GF/s\n"
                << "  Compute-only: " << compute_time_sec_max << " s, "
                << gflops_compute    << " GF/s\n";
  }
  // --------------------------------------------------------

  // Copy final result back to host (excluded from timing above)
  cudaMemcpyAsync(C.data.data(), dC, C_size, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Cleanup timing handles
  cudaEventDestroy(comp_start);
  cudaEventDestroy(comp_stop);

  // Clean up device resources
  cudaFree(dC);
  cudaStreamDestroy(stream);

  // Verify result
  if (do_verify) {
      std::vector<double> fullC;
      gather_matrix(C, N, d, fullC, 0);
      verify_result(N, fullC);
  }
}

#endif
