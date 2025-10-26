#include "localmatrix.hpp"

static void matmul_into(const double* A, const double* B, double* C,
                        int m, int n, int k);
static void add_inplace(double* C, const double* X,
                        int rows, int cols);

// C = A * B
static void matmul_into(const double* A, const double* B, double* C,
                        int m, int n, int k)
{
    // zero C
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            C[i * n + j] = 0.0;

    // C += A * B
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int kk = 0; kk < k; ++kk) {
                sum += A[i * k + kk] * B[kk * n + j];
            }
            C[i * n + j] += sum;
        }
    }
}

// C += X  (rows x cols), row-major
static void add_inplace(double* C, const double* X,
                        int rows, int cols)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            C[i * cols + j] += X[i * cols + j];
}

// Write your local DGEMM usign OpenMP
void local_dgemm_cpu(LocalMatrix& C, const double* Arow, int Am, int An,
                                    const double* Bcol, int Bm, int Bn)
{
    std::vector<double> tmp(static_cast<size_t>(Am) * Bn);
    matmul_into(Arow, Bcol, tmp.data(), Am, Bn, An);
    add_inplace(C.data.data(), tmp.data(), Am, Bn);
}