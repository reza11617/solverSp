#include <iostream>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cusolverSp.h>

#include "SolverSp/SolverSp.h"

int main()
{
  // define the CSR matrix
  int nRow = 4; int nCol = 4; int N = 4; int nnz = 5;
  float        *A;            cudaMallocManaged(&A, nnz*sizeof(float));
  unsigned int *A_RowIndices; cudaMallocManaged(&A_RowIndices, (nRow+1)*sizeof(unsigned int)); 
  unsigned int *A_ColIndices; cudaMallocManaged(&A_ColIndices, nnz*sizeof(unsigned int));
  float        *F;            cudaMallocManaged(&F, nRow*sizeof(float));
  float        *x;            cudaMallocManaged(&x, nRow*sizeof(float));
  
  A_RowIndices[0] = 0;
  A_RowIndices[1] = 1;
  A_RowIndices[2] = 3;
  A_RowIndices[3] = 4;
  A_RowIndices[4] = 5;
  
  A[0] = 34.0; A_ColIndices[0] = 0;
  A[1] = 12.0; A_ColIndices[1] = 0;
  A[2] = 41.0; A_ColIndices[2] = 1;
  A[3] =  1.0; A_ColIndices[3] = 2;
  A[4] =  1.0; A_ColIndices[4] = 3;
  
  F[0] = 100.0; F[1] = 200.0; F[2] = 400.0; F[3] = 500.0;

  SolverSp(A, A_RowIndices, A_ColIndices, nnz, N, F, x);
  
  printf("Showing the results...\n");
  for (int i = 0; i < N; i++)   std::cout<< "x[" << i << "]= " << x[i] << std::endl;
  for (int i = 0; i < N; i++)   std::cout<< "F[" << i << "]= " << F[i] << std::endl;
  for (int i = 0; i < nnz; i++) std::cout<< "A[" << i << "]= " << A[i] << std::endl;
}