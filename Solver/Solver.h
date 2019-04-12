#pragma once 

#include <iostream>
#include "ConjugateGradient.h"

class Solver {
private:

public:
  Solver(int N, int nz, double * val, int *rowPtr, int *colIndex, double * x, double* rhs);
  ~Solver();
private:

};