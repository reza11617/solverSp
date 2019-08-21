#include "Solver.h"

Solver::Solver(int N, int nz, double * val, int *rowPtr, int *colIndex, double * x, double* rhs) 
{
    double *rhs_dummy; // To not change the rhs vector
    cudaMallocManaged(&rhs_dummy, N*sizeof(double));
    cudaMemcpy(rhs_dummy, rhs, N*sizeof(double),cudaMemcpyDeviceToDevice);

/*
    INFO("Starting to solve ...");
    INFO("Cholesky Decomposition Selected");
    cudaMemset(x, 0, N*sizeof(double));
    //GPU::Cholesky(N, nz, val, rowPtr, colIndex, x, rhs_dummy, tol);
    INFO("Conjugate Gradient Selected (GPU)");
    cudaMemset(x, 0, N*sizeof(double));
    cudaMemcpy(rhs_dummy, rhs, N*sizeof(double),cudaMemcpyDeviceToDevice);
    //GPU::ConjugateGradient(N, nz, val, rowPtr, colIndex, x, rhs_dummy, tol);
    INFO("Conjugate Gradient Selected (SingleCPU)");
    cudaMemset(x, 0, N*sizeof(double));
    cudaMemcpy(rhs_dummy, rhs, N*sizeof(double),cudaMemcpyDeviceToDevice);
    //SingleCPU::ConjugateGradient(N, nz, val, rowPtr, colIndex, x, rhs_dummy, tol);
    INFO("Conjugate Gradient Precondition Selected (LU) (GPU)");
    cudaMemset(x, 0, N*sizeof(double));
    cudaMemcpy(rhs_dummy, rhs, N*sizeof(double),cudaMemcpyDeviceToDevice);
    GPU::ConjugateGradientPrecondLU(N, nz, val, rowPtr, colIndex, x, rhs_dummy, tol);
    cudaFree(rhs_dummy);
*/
    INFO("Conjugate Gradient Precondition Selected (CHOL) (GPU)");
    cudaMemset(x, 0, N*sizeof(double));
    //cudaMemcpy(rhs_dummy, rhs, N*sizeof(double),cudaMemcpyDeviceToDevice);
    GPU::ConjugateGradientPrecondChol(N, nz, val, rowPtr, colIndex, x, rhs_dummy, tol);
    cudaFree(rhs_dummy);
}

Solver::~Solver() {

}