#include "SolverSp.h"

SolverSp::SolverSp(float* A, unsigned int* rp, unsigned int* ci, unsigned int nz, unsigned int n, float* F, float* x)
  : sparseMatrix(A), rowPtr((int*) rp), colIndices((int*) ci), nnz((int) nz), N((int)n), rightHandSideVector(F), leftHandSideVector(x)
{
  tolerance = 0.000001;
  reorder = 0;
  if (SolverSpChol(sparseMatrix, rowPtr, colIndices) != -1) {
    printf("[Warning]: Matrix was not positive definitive trying to expand the matix it may take longer...\n");
    fixMatrices(); // try another method
  }
}
SolverSp::~SolverSp() {
  
}


int SolverSp::SolverSpChol(float* sp, int*rp, int*ci) {
  // --- Start the cuda sparse
  cusparseHandle_t handle; cusparseCreate(&handle);
  // --- Descriptor for sparse matrix A
  cusparseMatDescr_t descrA;      (cusparseCreateMatDescr(&descrA));
  cusparseSetMatType      (descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase (descrA, CUSPARSE_INDEX_BASE_ZERO);
  // --- CUDA solver initialization
  cusolverSpHandle_t solver_handle;
  cusolverSpCreate(&solver_handle);
  //
  int singularity;
  cusolverSpScsrlsvcholHost(solver_handle, N, nnz, descrA,
  			    sp, rp, ci,
  			    rightHandSideVector, tolerance, reorder,
  			    leftHandSideVector, &singularity);
  return singularity;
}

void SolverSp::SolverSpQR(float* sp, int*rp, int*ci) {
  cusparseHandle_t handle_n; cusparseCreate(&handle_n);
  // --- Descriptor for sparse matrix A
  cusparseMatDescr_t descrA_n;      (cusparseCreateMatDescr(&descrA_n));
  cusparseSetMatType      (descrA_n, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase (descrA_n, CUSPARSE_INDEX_BASE_ZERO);
  // --- CUDA solver initialization
  cusolverSpHandle_t solver_handle_n;
  cusolverSpCreate(&solver_handle_n);
  // --- start solving
  int rankA;
  int *p = new int[N];
  
  float min_norm;
  cusolverSpScsrlsqvqrHost(solver_handle_n, N, N, nnz, descrA_n,
  			   sp, rp, ci, rightHandSideVector,
  			   tolerance, &rankA, leftHandSideVector, p, &min_norm);
  delete[] p;
}

void SolverSp::fixMatrices() {
  int* rowPtr_n; cudaMallocManaged(&rowPtr_n, (N+1)*sizeof(int)); rowPtr_n[0] = 0;
  // -- find the new size

  for (int i = 0; i < nnz; i++) {
    rowPtr_n[colIndices[i]+1] = rowPtr_n[colIndices[i]+1] + 1;
  }
  for (int i = 1; i <= N; i++) {
    rowPtr_n[i] = (rowPtr[i] - rowPtr[i-1]) + (rowPtr_n[i]-1) + rowPtr_n[i-1]; 
  }
  nnz = rowPtr_n[N];
  // -- alocate new matrix variables
  float* A_n; int* colIndices_n;
  cudaMallocManaged(&A_n, nnz*sizeof(float));
  cudaMallocManaged(&colIndices_n, nnz*sizeof(int));
  // -- fix the new matrix
  int counter = 0;
  int counter_n;
  int e;
  for (int i = 0; i < N; i++) {
    counter_n =  rowPtr_n[i];
    for (int c = rowPtr[i]; c < rowPtr[i+1]; c++) {
      A_n[counter_n] = sparseMatrix[c];
      colIndices_n[counter_n] = colIndices[c];
      counter_n++;
    }
    e = i;
    for (int c = rowPtr[e+1]; c < rowPtr[N]; c++) {
      if (c >= rowPtr[e+1]) {e = e+1;}
      if (colIndices[c] == i) {
	A_n[counter_n] = sparseMatrix[c];
	
	colIndices_n[counter_n] = e;
	counter_n++;
      }
    }
  }
  // -- solve with the new matrix
  SolverSpQR(A_n, rowPtr_n, colIndices_n);
  // -- delete new variables for the new matrix 
  cudaFree(A_n);
  cudaFree(rowPtr_n);
  cudaFree(colIndices_n);
}
