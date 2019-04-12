#include "Solver.h"

Solver::Solver(int N, int nz, double * val, int *rowPtr, int *colIndex, double * x, double* rhs) 
{
    double *rhs_dummy; // To not change the rhs vector
    cudaMallocManaged(&rhs_dummy, N*sizeof(double));
    cudaMemcpy(rhs_dummy, rhs, N*sizeof(double),cudaMemcpyDeviceToDevice );

    std::cout<<"Starting to solve ...\n";
    std::cout<<"Conjugate Gradient Selected\n";
    ConjugateGradient(N, nz, val, rowPtr, colIndex, x, rhs_dummy);
        
    cudaFree(rhs_dummy);
}

Solver::~Solver() {

}