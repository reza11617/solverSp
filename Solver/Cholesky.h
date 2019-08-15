#pragma once 

#include <iostream>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cusolverSp.h>

class Cholesky {
private:


public:
  Cholesky(int N, int nz, double * val, int *rowPtr, int *colIndex, double * x, double* rhs, const double tol);
  ~Cholesky();
private:

};