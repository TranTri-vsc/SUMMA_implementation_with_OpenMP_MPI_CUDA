#include "verify.hpp"
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstring>

void gather_matrix(const LocalMatrix& C, int N, const Dist2D& d,
                   std::vector<double>& fullC, int root)
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int lrows = C.l_rows;
    const int lcols = C.l_cols;
    const int r0 = C.row_offset;
    const int c0 = C.col_offset;
    const int elems = lrows * lcols;

    const int META_TAG = 1001;
    const int DATA_TAG = 1002;

    if (rank == root) {
        // Root owns the final dense matrix
        fullC.assign(static_cast<size_t>(N) * N, 0.0);

        // Helper to copy a (h × w) tile stored row-major at 'tile'
        auto place_tile = [&](const double* tile, int h, int w, int row_base, int col_base) {
            for (int i = 0; i < h; ++i) {
                std::memcpy(fullC.data() + (static_cast<size_t>(row_base + i) * N + col_base),
                            tile + static_cast<size_t>(i) * w,
                            static_cast<size_t>(w) * sizeof(double));
            }
        };

        // 1) Place root's own tile directly
        place_tile(C.data.data(), lrows, lcols, r0, c0);

        // 2) Receive from every other rank
        for (int src = 0; src < size; ++src) {
            if (src == root) continue;

            // Receive metadata
            int meta[4] = {0, 0, 0, 0};
            MPI_Recv(meta, 4, MPI_INT, src, META_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            const int mr = meta[0];
            const int nr = meta[1];
            const int r00 = meta[2];
            const int c00 = meta[3];

            // Receive tile data
            std::vector<double> tile(static_cast<size_t>(mr) * nr);
            MPI_Recv(tile.data(), mr * nr, MPI_DOUBLE, src, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Stitch into fullC
            place_tile(tile.data(), mr, nr, r00, c00);
        }
    } else {
        // Non-root: send metadata then the raw tile buffer
        int meta[4] = { lrows, lcols, r0, c0 };
        MPI_Send(meta, 4, MPI_INT, root, META_TAG, MPI_COMM_WORLD);
        MPI_Send(const_cast<double*>(C.data.data()), elems, MPI_DOUBLE, root, DATA_TAG, MPI_COMM_WORLD);

        // Non-root does not own the assembled matrix
        fullC.clear();
    }
}

void verify_result(int N, const std::vector<double>& fullC)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) return; // only root prints

    const double tol = 1e-5;

    // Precompute sums in double to avoid integer overflow
    const double Nd = static_cast<double>(N);
    const double S1 = Nd * (Nd - 1.0) * 0.5;                     // sum_{k=0}^{N-1} k
    const double S2 = (Nd - 1.0) * Nd * (2.0 * Nd - 1.0) / 6.0;  // sum_{k=0}^{N-1} k^2

    double max_diff = 0.0;

    // One pass: compute expected C(i,j) analytically and compare
    #pragma omp parallel for reduction(max:max_diff) schedule(static) collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const double id = static_cast<double>(i);
            const double jd = static_cast<double>(j);

            // Expected C(i,j)
            const double expected = (id - jd) * S1 - id * jd * Nd + S2;

            const size_t idx = static_cast<size_t>(i) * N + j;
            const double diff = std::fabs(fullC[idx] - expected);
            if (diff > max_diff) max_diff = diff;
        }
    }

    std::cout << "VERIFY RESULT:  max C diff = " << max_diff << '\n';
    if (max_diff <= tol) {
        std::cout << "RESULT: PASS (tolerance = " << tol << ")\n";
    } else {
        std::cout << "RESULT: FAIL (tolerance = " << tol << ")\n";
    }
}