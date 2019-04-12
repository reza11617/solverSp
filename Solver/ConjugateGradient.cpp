#include "ConjugateGradient.h"

ConjugateGradient::ConjugateGradient(int N, int nz, double * val, int *rowPtr, int *colIndex, double * x, double* rhs)
{
    const int max_iter = 10000;
    const double tol = 0.00001d;
    int k = 0;
    double alpha, beta, alpham1, dot;
    double a, b, na, r0, r1;
    double *p, *Ax;

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.;

    cudaMallocManaged(&p, N*sizeof(double));
    cudaMallocManaged(&Ax, N*sizeof(double));

    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);


    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);


    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);


    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_TRANSPOSE, N, N, nz, &alpha, descr, val, rowPtr, colIndex, x, &beta, Ax);
    cublasDaxpy(cublasHandle, N, &alpham1, Ax, 1, rhs, 1);
    cublasStatus = cublasDdot(cublasHandle, N, rhs, 1, rhs, 1, &r1);


    k = 1;

    while (r1 > tol*tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            cublasStatus = cublasDscal(cublasHandle, N, &b, p, 1);
            cublasStatus = cublasDaxpy(cublasHandle, N, &alpha, rhs, 1, p, 1);
        }
        else
        {
            cublasStatus = cublasDcopy(cublasHandle, N, rhs, 1, p, 1);
        }

        cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, N, N, nz, &alpha, descr, val, rowPtr, colIndex, p, &beta, Ax);
        cublasStatus = cublasDdot(cublasHandle, N, p, 1, Ax, 1, &dot);
        a = r1 / dot;

        cublasStatus = cublasDaxpy(cublasHandle, N, &a, p, 1, x, 1);
        na = -a;
        cublasStatus = cublasDaxpy(cublasHandle, N, &na, Ax, 1, rhs, 1);

        r0 = r1;
        cublasStatus = cublasDdot(cublasHandle, N, rhs, 1, rhs, 1, &r1);
        cudaDeviceSynchronize();
        std::cout<<"iteration = "<< k <<", residual = "<<sqrt(r1)<<"\n";
        k++;
    }


    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    cudaFree(p);
    cudaFree(Ax);
}

ConjugateGradient::~ConjugateGradient() {

}