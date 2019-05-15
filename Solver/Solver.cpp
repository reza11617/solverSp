#include "Solver.h"

Solver::Solver(int N, int nz, double * val, int *rowPtr, int *colIndex, double * x, double* rhs) 
{
    double *rhs_dummy; // To not change the rhs vector
    cudaMallocManaged(&rhs_dummy, N*sizeof(double));
    cudaMemcpy(rhs_dummy, rhs, N*sizeof(double),cudaMemcpyDeviceToDevice);

    std::cout<<"Starting to solve ...\n";
    std::cout<<"Cholesky Decomposition Selected\n";
    cudaMemset(x, 0, N*sizeof(double));
    Cholesky(N, nz, val, rowPtr, colIndex, x, rhs_dummy, tol);
    std::cout<<"Conjugate Gradient Selected\n";
    cudaMemset(x, 0, N*sizeof(double));
    cudaMemcpy(rhs_dummy, rhs, N*sizeof(double),cudaMemcpyDeviceToDevice);
    ConjugateGradient(N, nz, val, rowPtr, colIndex, x, rhs_dummy, tol);
    std::cout<<"Conjugate Gradient Precondition Selected\n";
    cudaMemset(x, 0, N*sizeof(double));
    cudaMemcpy(rhs_dummy, rhs, N*sizeof(double),cudaMemcpyDeviceToDevice);
    ConjugateGradientPrecond(N, nz, val, rowPtr, colIndex, x, rhs_dummy, tol);
    cudaFree(rhs_dummy);
        
}

Solver::~Solver() {

}