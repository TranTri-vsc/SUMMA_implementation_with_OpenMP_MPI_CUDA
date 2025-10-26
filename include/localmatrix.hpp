#pragma once
#include <vector>
#include <algorithm> // for std::fill
#include "common.hpp"

// 2D-block distributed dense matrix, local tile stored row-major in `data`.
class LocalMatrix {
public:
    int g_rows{0}, g_cols{0};   // global rows and global cols
    int l_rows{0}, l_cols{0};   // local rows and local cols
    Dist2D grid;                // process grid info
    std::vector<double> data;   // local tile data
    int row_offset{0}, col_offset{0}; // global row and col offset of local tile

    LocalMatrix(int g_m, int g_n, const Dist2D& d) 
        : g_rows(g_m), g_cols(g_n), grid(d)
    {
        // initialize process grid info
        std::vector<int> row_sz, row_offs, col_sz, col_offs;
        
        // Split global matrix sizes to local sizes
        split_sizes(g_rows, grid.P, row_sz, row_offs);
        split_sizes(g_cols, grid.P, col_sz, col_offs);

        // l_rows and l_cols are the sizes of the local tile
        l_rows = row_sz[grid.myr];
        l_cols = col_sz[grid.myc];

        // row_offset and col_offset are the global indices of the first element in the local tile
        row_offset = row_offs[grid.myr];
        col_offset = col_offs[grid.myc];

        data.assign(l_rows * l_cols, 0.0);
    }

    // double& operator()(int i,int j){ return data[i*l_cols+j]; }
    // const double& operator()(int i,int j) const { return data[i*l_cols+j]; }

    void zero() {
        std::fill(data.begin(), data.end(), 0.0);
    }

    // Deterministic initializers (simple integers; no comm needed)
    void initialize_A()
    {
        for (int i = 0; i < l_rows; ++i) 
        {
            const int i_val = row_offset + i;

            for (int j = 0; j < l_cols; ++j) 
            {
                const int j_val = col_offset + j;
                data[i * l_cols + j] = static_cast<double>(i_val + j_val);
            }
        }
    }

    void initialize_B() 
    {
        for (int i = 0; i < l_rows; ++i) 
        {
            const int i_val = row_offset + i;

            for (int j = 0; j < l_cols; ++j) 
            {
                const int j_val = col_offset + j;
                data[i * l_cols + j] = static_cast<double>(i_val - j_val);
            }
        }
    }
};
