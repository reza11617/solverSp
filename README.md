# solverSp
This class tries to solve positive definite matrices in CSR sparse format. 

A * x = F

As it is shown in the main file all parameters shall be defined on unified memory

```cpp
float           *A;            cudaMallocManaged(&A, nnz*sizeof(float));
unsigned int    *A_RowIndices; cudaMallocManaged(&A_RowIndices, (nRow+1)*sizeof(unsigned int));
unsigned int    *A_ColIndices; cudaMallocManaged(&A_ColIndices, nnz*sizeof(unsigned int));
float           *F;            cudaMallocManaged(&F, nRow*sizeof(float));
float           *x;            cudaMallocManaged(&x, nRow*sizeof(float));
```

the sample matrix is:  

![equation](http://mathurl.com/y8rdkmb6.png)

the right hand side vector is:

![equation](http://mathurl.com/yaeessnm.png)


The class tries to solve the linear problem using [cholesky decomposition](https://docs.nvidia.com/cuda/cusolver/index.html#cusolver-lt-t-gt-csrlsvchol)
