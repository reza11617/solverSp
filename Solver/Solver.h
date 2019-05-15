#pragma once 

#include <iostream>
#include "ConjugateGradient.h"
#include "ConjugateGradientPrecond.h"
#include "Cholesky.h"

class Solver {
protected:
    static constexpr double tol = 0.00001;
public:
  Solver(int N, int nz, double * val, int *rowPtr, int *colIndex, double * x, double* rhs);
  ~Solver();
private:

};