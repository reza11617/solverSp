#include "ConjugateGradient.h"

ConjugateGradient::ConjugateGradient(int N, int nz, double * val, int *rowPtr, int *colIndex, double * x, double* rhs, const double tol)
{
    const int max_iter = 10000;
    int k = 0;
    double alpha, nalpha,beta, dot;
    double a, b, na, r0, r1;
    double *p, *omega;;

    const double floatone = 1.0;
    const double floatzero = 0.0;


    alpha = 1.0;
    beta = 0.0;
    r0 = 0.;

    cudaMallocManaged(&p, N*sizeof(double));
    cudaMallocManaged(&omega, N*sizeof(double));

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


    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    cublasDdot(cublasHandle, N, rhs, 1, rhs, 1, &r1);

        while (r1 > tol*tol && k <= max_iter)
    {
        k++;

        if (k == 1)
        {
            cublasDcopy(cublasHandle, N, rhs, 1, p, 1);
        }
        else
        {
            beta = r1/r0;
            cublasDscal(cublasHandle, N, &beta, p, 1);
            cublasDaxpy(cublasHandle, N, &floatone, rhs, 1, p, 1) ;
        }

        cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &floatone, descr, val, rowPtr, colIndex, p, &floatzero, omega);
        cublasDdot(cublasHandle, N, p, 1, omega, 1, &dot);
        alpha = r1/dot;
        cublasDaxpy(cublasHandle, N, &alpha, p, 1, x, 1);
        nalpha = -alpha;
        cublasDaxpy(cublasHandle, N, &nalpha, omega, 1, rhs, 1);
        r0 = r1;
        cublasDdot(cublasHandle, N, rhs, 1, rhs, 1, &r1);
    }

    printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));



    /*
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
        k++;
    }

    std::cout<<"iteration = "<< k <<", residual = "<<sqrt(r1)<<"\n";
    */
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    cudaFree(p);
}

ConjugateGradient::~ConjugateGradient() {

}