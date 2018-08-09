#ifndef SOLVERSP_H
#define SOLVERSP_H

#include <iostream>

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cusolverSp.h>

// This class solves the symmetric sparse systems A(N*N) * x(N) = F(N)
class SolverSp {
private:
  int* rowPtr;
  int* colIndices;
  float* sparseMatrix;  // matrix has to be CSR
  float* rightHandSideVector;
  float* leftHandSideVector;
  int nnz; // number of non-zeros
  int N; // size of the matrix and vectors
  float tolerance; // tolerance to decide singularity
public:
  SolverSp(float*, unsigned int*, unsigned int*, unsigned int, unsigned int, float*, float*);
  //SolverSp(float*, unsigned int*, unsigned int*, unsigned int, unsigned int,float*, float*, float);
  ~SolverSp();
private:
  int SolverSpChol();
  void SolverSpQR();
  void fixMatrices();
};
#endif
