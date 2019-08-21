/////////////////////////////////////////////////////////////////////
// Developer::Reza Rahimi
// Date:: 18/08/19
// title:: multipling a CSR matrix to a vector 
/////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <cmath>

#include "SparseMatrixOperations.h"

static constexpr double tol = 0.00001;


int main() 
{
    // Example CSR matrix
    int rowPtr[] = {0,3,6,9,12};
    int colIndex[] = {0,1,2,0,1,3,0,2,3,1,2,3};
    double val[] = {-4,1,1,1,-4,1,1,-4,1,1,1,-4};
    // Sample left hand side and right hand side vectors
    double rhs[] = {-1,-1,0,0};
    double x[] = {0.0, 0.0, 0.0, 0.0};
    // ------------- conjugate gradient --------------
    SparseMatrixOperations<int, double> sp;
    const int max_iter = 1000;
    int k = 0;
    int N = 4; // number of rowPtrs
    double r0 = 0.0, r1, dot, alpha, nalpha;
    double beta = 0.0;
    double doubleOne = 1.0;
    double* p = new double[N];
    double* omega = new double[N];
    // dot prodoct
    sp.dot(N,rhs,rhs, &r1);
    while (r1 > tol*tol && k <= max_iter)
    {
        k++;
        if (k == 1) { sp.copy(N, rhs, p); }
        else 
        {
            beta = r1/r0;
            sp.scale(N,beta,p);
            sp.axpy(N, doubleOne, rhs, p);
        }
        sp.csrmv(N, rowPtr, colIndex, val, p, omega); 
        sp.dot(N,p,omega, &dot);
        alpha = r1/dot;
        sp.axpy(N, alpha, p, x);
        nalpha = -alpha;
        sp.axpy(N, nalpha, omega, rhs);
        r0 = r1;
        sp.dot( N, rhs, rhs, &r1);
    }

    std::string report = "  iteration = " + std::to_string(k) + " residual = " + std::to_string(std::sqrt(r1));
    std::cout<<report<<"\n";



    // check the results
    for (int i = 0; i < 4; i++) {std::cout << "F[" << i << "]= " << x[i] << std::endl;}
    // deletes
    delete[] p;
    delete[] omega;
} 
