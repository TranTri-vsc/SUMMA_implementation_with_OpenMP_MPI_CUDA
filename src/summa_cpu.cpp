#include "summa.hpp"
#include "localmatrix.hpp"
#include "verify.hpp"
#include <vector>
#include <iostream>

extern void local_dgemm_cpu(LocalMatrix& C, const double* Arow, int Am, int An,
                            const double* Bcol, int Bm, int Bn);

void run_summa_cpu(int N,const Dist2D& d, bool do_verify){
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

  // Partitioning info (sizes per process block)
  std::vector<int> row_sizes, row_offsets;
  std::vector<int> col_sizes, col_offsets;
  split_sizes(N, d.P, row_sizes, row_offsets);
  split_sizes(N, d.P, col_sizes, col_offsets);

  // Local tile dimensions
  const int local_rows = A.l_rows;   // rows owned by this rank
  const int local_cols = B.l_cols;   // cols owned by this rank

  // -------------------- TIMING SETUP --------------------
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();      // end-to-end (includes MPI)
  double compute_sec_accum = 0.0;     // sum of local compute-only segments
  // ------------------------------------------------------

  // SUMMA over k-blocks (shared dimension panels)
  for (int block_k = 0; block_k < d.P; block_k++) {
    const int k_dim = col_sizes[block_k];  // width of A-panel == height of B-panel

    // A-panel: (local_rows x k_dim), broadcast across row
    const double* A_panel = nullptr;
    std::vector<double> A_recv_buffer;

    if (d.myc == block_k) {
      A_panel = A.data.data();
    } else {
      A_recv_buffer.resize(static_cast<size_t>(local_rows) * k_dim);
      A_panel = A_recv_buffer.data();
    }
    MPI_Bcast(const_cast<double*>(A_panel),
              local_rows * k_dim, MPI_DOUBLE,
              block_k, d.row_comm);

    // B-panel: (k_dim x local_cols), broadcast down column
    const double* B_panel = nullptr;
    std::vector<double> B_recv_buffer;

    if (d.myr == block_k) {
      B_panel = B.data.data();
    } else {
      B_recv_buffer.resize(static_cast<size_t>(k_dim) * local_cols);
      B_panel = B_recv_buffer.data();
    }
    MPI_Bcast(const_cast<double*>(B_panel),
              k_dim * local_cols, MPI_DOUBLE,
              block_k, d.col_comm);

    // ---- Local compute: C += A_panel * B_panel ----
    const double c0 = MPI_Wtime();
    local_dgemm_cpu(C, A_panel, local_rows, k_dim,
                       B_panel, k_dim, local_cols);
    const double c1 = MPI_Wtime();
    compute_sec_accum += (c1 - c0);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double t1 = MPI_Wtime();

  // -------------------- GFLOP/s REPORT --------------------
  const double loop_time_sec_local    = t1 - t0;          // end-to-end
  const double compute_time_sec_local = compute_sec_accum; // compute-only

  double loop_time_sec_max    = 0.0;
  double compute_time_sec_max = 0.0;

  MPI_Reduce((void*)&loop_time_sec_local,    (void*)&loop_time_sec_max,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce((void*)&compute_time_sec_local, (void*)&compute_time_sec_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  const double global_flops = 2.0 * static_cast<double>(N) * N * N;

  if (rank == 0) {
    const double gflops_end_to_end = (global_flops / loop_time_sec_max)    / 1e9;
    const double gflops_compute    = (global_flops / compute_time_sec_max) / 1e9;

    std::cout << "[CPU] N=" << N << " P=" << d.P <<"x" << d.P << " ranks=" << world << "\n"
              << "  End-to-end  : " << loop_time_sec_max    << " s, "
              << gflops_end_to_end << " GF/s\n"
              << "  Compute-only: " << compute_time_sec_max << " s, "
              << gflops_compute    << " GF/s\n";
  }
  // ------------------------------------------------------

  // Verify result
  if (do_verify) {
    std::vector<double> fullC;
    gather_matrix(C, N, d, fullC, 0);
    verify_result(N, fullC);
  }
}
