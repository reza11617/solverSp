#pragma once 

#include <iostream>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

class ConjugateGradient {
private:

public:
  ConjugateGradient(int N, int nz, double * val, int *rowPtr, int *colIndex, double * x, double* rhs, const double tol);
  ~ConjugateGradient();
private:

};