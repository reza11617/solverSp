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
/* genLaplace: Generate a matrix representing a second order, regular, Laplacian operator on a 2d domain in Compressed Sparse Row format*/
void genLaplace(int *row_ptr, int *col_ind, double *val, int M, int N, int nz, double *rhs)
{
    assert(M==N);
    int n=(int)sqrt((double)N);
    assert(n*n==N);
    printf("laplace dimension = %d\n", n);
    int idx = 0;

    // loop over degrees of freedom
    for (int i=0; i<N; i++)
    {
        int ix = i%n;
        int iy = i/n;

        row_ptr[i] = idx;

        // up
        if (iy > 0)
        {
            val[idx] = 1.0;
            col_ind[idx] = i-n;
            idx++;
        }
        else
        {
            rhs[i] -= 1.0;
        }

        // left
        if (ix > 0)
        {
            val[idx] = 1.0;
            col_ind[idx] = i-1;
            idx++;
        }
        else
        {
            rhs[i] -= 0.0;
        }

        // center
        val[idx] = -4.0;
        col_ind[idx]=i;
        idx++;

        //right
        if (ix  < n-1)
        {
            val[idx] = 1.0;
            col_ind[idx] = i+1;
            idx++;
        }
        else
        {
            rhs[i] -= 0.0;
        }

        //down
        if (iy  < n-1)
        {
            val[idx] = 1.0;
            col_ind[idx] = i+n;
            idx++;
        }
        else
        {
            rhs[i] -= 0.0;
        }

    }

    row_ptr[N] = idx;

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
    std::cout<<rowPtr[N]<<"\n";
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
    int M, N, nz ;
    int *colIndex, *rowPtr;
    int k = 0;
    double *val, *x;
    double *rhs;
    M = N = 4;
    nz = 5*N-4*(int)sqrt((double)N);

    cudaMallocManaged(&colIndex, nz*sizeof(int));
    cudaMallocManaged(&rowPtr, (N+1)*sizeof(int));
    cudaMallocManaged(&val, nz*sizeof(double));
    cudaMallocManaged(&x, N*sizeof(double));
    cudaMallocManaged(&rhs, N*sizeof(double));

    genLaplace(rowPtr, colIndex,val, M, N, nz, rhs);

    PrintMatrix(rowPtr, colIndex, val, N, nz);

    // for (int i = 0; i<N; i++)
    //     std::cout<<rhs[i]<<"\n"; 

    Solver(N, nz, val, rowPtr, colIndex, x, rhs); 
    double rsum, diff, err = 0.0;

    for (int i = 0; i < N; i++)
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
    

     for (int i = 0; i<N; i++)
         std::cout<<x[i]<<"\n"; 

    cudaFree(colIndex);
    cudaFree(rowPtr);
    cudaFree(val);
    cudaFree(x);
    cudaFree(rhs);



    std::cout<<"Test Summary:  Error amount = "<< err <<"\n";
}
