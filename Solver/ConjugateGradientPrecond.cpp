#include "ConjugateGradientPrecond.h"

ConjugateGradientPrecond::ConjugateGradientPrecond(int N, int nz, double * val, int *rowPtr, int *colIndex, double * x, double* rhs, const double tol)
{
    /* Conjugate Gradient variable */
    const int max_iter = 10000;
    int k = 0;
    double alpha, beta, alpham1; 
    double r1;
    double numerator, denominator, nalpha;
    double *p, *y, *omega;

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;

    cudaMallocManaged(&p, N*sizeof(double));
    cudaMallocManaged(&y, N*sizeof(double));
    cudaMallocManaged(&omega, N*sizeof(double));

    /* Conjugate Gradient Precond variable */
    int nzILU0 = 2*N-1;
    double *valsILU0; cudaMallocManaged(&valsILU0, nz*sizeof(double));
    double *zm1; cudaMallocManaged(&zm1, N*sizeof(double));
    double *zm2; cudaMallocManaged(&zm2, N*sizeof(double));
    double *rm2; cudaMallocManaged(&rm2, N*sizeof(double));
    const double floatone = 1.0;
    const double floatzero = 0.0;
    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);
    
    /* create the analysis info object for the A matrix */
    cusparseSolveAnalysisInfo_t infoA = 0;
    cusparseStatus = cusparseCreateSolveAnalysisInfo(&infoA);

    /* Description of the A matrix*/
    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);

    /* Define the properties of the matrix */
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    cusparseStatus = cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            N, nz, descr, val, rowPtr, colIndex, infoA);
    checkCudaErrors(cusparseStatus);
    /* Copy A data to ILU0 vals as input*/
    cudaMemcpy(valsILU0, val, nz*sizeof(double), cudaMemcpyDeviceToDevice);

    /* generate the Incomplete LU factor H for the matrix A using cudsparseDcsrilu0 */
    cusparseStatus = cusparseDcsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, descr, valsILU0, rowPtr, colIndex, infoA);
    checkCudaErrors(cusparseStatus);
    /* Create info objects for the ILU0 preconditioner */
    cusparseSolveAnalysisInfo_t info_u;
    cusparseCreateSolveAnalysisInfo(&info_u);

    cusparseMatDescr_t descrL = 0;
    cusparseStatus = cusparseCreateMatDescr(&descrL);
    cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrL,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT); 

    cusparseMatDescr_t descrU = 0;
    cusparseStatus = cusparseCreateMatDescr(&descrU);
    cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrU,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, descrU, val, rowPtr, colIndex, info_u);

    cublasDdot(cublasHandle, N, rhs, 1, rhs, 1, &r1); 

    while (r1 > tol*tol && k <= max_iter)
    {
        // Forward Solve, we can re-use infoA since the sparsity pattern of A matches that of L
        cusparseStatus = cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &floatone, descrL,
                            valsILU0, rowPtr, colIndex, infoA, rhs, y);
        checkCudaErrors(cusparseStatus);
        cusparseStatus = cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &floatone, descrU,
                             valsILU0, rowPtr, colIndex, info_u, y, zm1);
        checkCudaErrors(cusparseStatus);
        k++;

        if (k == 1)
        {
            cublasDcopy(cublasHandle, N, zm1, 1, p, 1);
        }
        else
        {
            cublasDdot(cublasHandle, N, rhs, 1, zm1, 1, &numerator);
            cublasDdot(cublasHandle, N, rm2, 1, zm2, 1, &denominator);
            beta = numerator/denominator;
            cublasDscal(cublasHandle, N, &beta, p, 1);
            cublasDaxpy(cublasHandle, N, &floatone, zm1, 1, p, 1) ;
        }

        cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nzILU0, &floatone, descrU, val, rowPtr, colIndex, p, &floatzero, omega);
        cublasDdot(cublasHandle, N, rhs, 1, zm1, 1, &numerator);
        cublasDdot(cublasHandle, N, p, 1, omega, 1, &denominator);
        alpha = numerator / denominator;
        cublasDaxpy(cublasHandle, N, &alpha, p, 1, x, 1);
        cublasDcopy(cublasHandle, N, rhs, 1, rm2, 1);
        cublasDcopy(cublasHandle, N, zm1, 1, zm2, 1);
        nalpha = -alpha;
        cublasDaxpy(cublasHandle, N, &nalpha, omega, 1, rhs, 1);
        cublasDdot(cublasHandle, N, rhs, 1, rhs, 1, &r1);
    }

    std::cout<<"iteration = "<<k<<", residual = "<<sqrt(r1)<<"\n";

    cusparseDestroySolveAnalysisInfo(infoA);
    cusparseDestroySolveAnalysisInfo(info_u);

    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    cudaFree(p);
    cudaFree(y);
    cudaFree(p);
    cudaFree(omega);
    cudaFree(valsILU0);
    cudaFree(zm1);
    cudaFree(zm2);
    cudaFree(rm2);
}

ConjugateGradientPrecond::~ConjugateGradientPrecond(){

}