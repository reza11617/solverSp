#include <iostream>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cusolverSp.h>

#include "SolverSp/SolverSp.h"

int main()
{
  // -- define the CSR matrix A * x = F
  int nRow = 4; int nCol = 4; int N = 4; 
  float        *A;            
  unsigned int *A_RowIndices; 
  unsigned int *A_ColIndices; 
  float        *F;            
  float        *x;
  cudaMallocManaged(&F, nRow*sizeof(float));
  F[0] = 100.0; F[1] = 200.0; F[2] = 400.0; F[3] = 500.0;
  cudaMallocManaged(&x, nRow*sizeof(float));
  cudaMallocManaged(&A_RowIndices, (nRow+1)*sizeof(unsigned int));
  A_RowIndices[0] = 0;
  // -- A positive definite matrix
  /*
  int nnz = 5;
  cudaMallocManaged(&A, nnz*sizeof(float));
  cudaMallocManaged(&A_ColIndices, nnz*sizeof(unsigned int));
  A[0] = 34.0; A_ColIndices[0] = 0; A_RowIndices[1] = 1;
  A[1] = 12.0; A_ColIndices[1] = 0; 
  A[2] = 41.0; A_ColIndices[2] = 1; A_RowIndices[2] = 3;
  A[3] =  1.0; A_ColIndices[3] = 2; A_RowIndices[3] = 4;
  A[4] =  1.0; A_ColIndices[4] = 3; A_RowIndices[4] = 5;
  */
  // -- A symmetric matrix
  unsigned int nnz = 7;
  cudaMallocManaged(&A, nnz*sizeof(float));
  cudaMallocManaged(&A_ColIndices, nnz*sizeof(unsigned int));
  A[0] = 1.0; A_RowIndices[1] = 1; A_ColIndices[0] = 0;
  A[1] = 4.0;                      A_ColIndices[1] = 0;
  A[2] = 2.0; A_RowIndices[2] = 3; A_ColIndices[2] = 1;
  A[3] = 5.0;                      A_ColIndices[3] = 0;
  A[4] = 1.0; A_RowIndices[3] = 5; A_ColIndices[4] = 2;
  A[5] = 6.0;                      A_ColIndices[5] = 2;
  A[6] = 8.0; A_RowIndices[4] = 7; A_ColIndices[6] = 3;

  
  // -- solver call
  SolverSp(A, A_RowIndices, A_ColIndices, nnz, N, F, x);
  // -- print result
  printf("Showing the results...\n");
  for (int i = 0; i < N; i++)   std::cout<< "x[" << i << "]= " << x[i] << std::endl;
  for (int i = 0; i < N; i++)   std::cout<< "F[" << i << "]= " << F[i] << std::endl;
  for (int i = 0; i < nnz; i++) std::cout<< "A[" << i << "]= " << A[i] << std::endl;
  // -- Cuda free
  cudaFree(A);
  cudaFree(A_RowIndices);
  cudaFree(A_ColIndices);
  cudaFree(F);
  cudaFree(x);
}
