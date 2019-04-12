#include <iostream>

#include "Solver/Solver.h"

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper function CUDA error checking and initialization

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *rowPtr, int *colIndex, double *val, int N, int nz)
{
    rowPtr[0] = 0;
    for (int i = 1; i <= N; i++)
    {
        rowPtr[i] = rowPtr[i-1] + i;
        int counter = 0;
        for (auto j = rowPtr[i-1]; j < rowPtr[i]; j++)
        {
            colIndex[j] = counter++;
            val[j] = (double)rand()/RAND_MAX + 10.0d;
        }

    }
}

void PrintMatrix(int *rowPtr, int *colIndex, double *val, int N, int nz)
{
    for (auto i = 0; i < N; i++)
    {
        std::cout<<rowPtr[i];
        for (auto j = rowPtr[i]; j < rowPtr[i+1]; j++)
        {
            std::cout<<"\t"<<colIndex[j]<<"\t"<<val[j]<<"\n";
        }
    }
}

int main(int argc, char **argv)
{
     // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;
    int devID = findCudaDevice(argc, (const char **)argv);

    if (devID < 0)
    {
        std::cout<<"exiting...\n";
        exit(EXIT_SUCCESS);
    }
    cudaGetDeviceProperties(&deviceProp, devID);
    // Statistics about the GPU device
    std::cout<<"> GPU device has " << deviceProp.multiProcessorCount << " Multi-Processors, SM "<< deviceProp.major <<"."<< deviceProp.minor <<" compute capabilities\n\n";

    int M = 0, N = 300, nz ;
    int *colIndex, *rowPtr;
    int k = 0;
    double *val, *x;
    double *rhs;

    cudaMallocManaged(&colIndex, nz*sizeof(int));
    cudaMallocManaged(&rowPtr, (N+1)*sizeof(int));
    cudaMallocManaged(&val, nz*sizeof(double));
    cudaMallocManaged(&x, N*sizeof(double));
    cudaMallocManaged(&rhs, N*sizeof(double));

    genTridiag(rowPtr, colIndex, val, N, nz);


    for (int i = 0; i < N; i++)
    {
        rhs[i] = 1.0;
        x[i] = 0.0;
    }
    Solver(N, nz, val, rowPtr, colIndex, x, rhs);
    double rsum, diff, err = 0.0;

    for (int i = N-1; i < N; i++)
    {
        rsum = 0.0;

        for (int j = rowPtr[i]; j < rowPtr[i+1]; j++)
        {
            rsum += val[j]*x[colIndex[j]];
        }

        diff = fabs(rsum - rhs[i]);

        if (diff > err)
        {
            err = diff;
        }
    }
    
/*     PrintMatrix(rowPtr, colIndex, val, N, nz);
    for (int i = 0; i<N; i++)
        std::cout<<x[i]<<"\n"; 

   */

    cudaFree(colIndex);
    cudaFree(rowPtr);
    cudaFree(val);
    cudaFree(x);
    cudaFree(rhs);



    std::cout<<"Test Summary:  Error amount = "<< err <<"\n";
    //exit((k <= max_iter) ? 0 : 1);
    


}
