#pragma once
#include "localmatrix.hpp"
#include <vector>

void gather_matrix(const LocalMatrix& C, int N, const Dist2D& d,
                   std::vector<double>& fullC, int root=0);
void verify_result(int N, const std::vector<double>& fullC);
