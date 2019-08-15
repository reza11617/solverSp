#pragma once 

#include <iostream>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper function CUDA error checking and initialization

class ConjugateGradientPrecond {
private:

public:
    ConjugateGradientPrecond(int N, int nz, double * val, int *rowPtr, int *colIndex, double * x, double* rhs, const double tol);
    ~ConjugateGradientPrecond();

};