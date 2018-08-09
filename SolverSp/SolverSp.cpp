#include "SolverSp.h"

SolverSp::SolverSp(float* A, unsigned int* rp, unsigned int* ci, unsigned int nz, unsigned int n, float* F, float* x)
  : sparseMatrix(A), rowPtr((int*) rp), colIndices((int*) ci), nnz((int) nz), N((int)n), rightHandSideVector(F), leftHandSideVector(x)
{
  tolerance = 0.000001;
  if (SolverSpChol() != -1) {
    printf("[Warning]: Matrix was not positive definitive trying QR decomposition it may take longer...\n");
    //SolverSpQR(); // try another method
  }
}
SolverSp::~SolverSp() {
  
}


int SolverSp::SolverSpChol() {
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
  int reorder = 0;
  int singularity;
  cusolverSpScsrlsvcholHost(solver_handle, N, nnz, descrA,
  			    sparseMatrix, rowPtr, colIndices,
  			    rightHandSideVector, tolerance, reorder,
  			    leftHandSideVector, &singularity);
  return singularity;
}

void SolverSp::SolverSpQR() {
  fixMatrices();
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
  //cusolverSpScsrlsqvqrHost(solver_handle_n, N, N, nnz, descrA_n,
  //			   sparseMatrix, rowPtr, colIndices, rightHandSideVector,
  //			   tolerance, &rankA, leftHandSideVector, p, &min_norm);
  delete[] p;
}

void SolverSp::fixMatrices() {
  
}
