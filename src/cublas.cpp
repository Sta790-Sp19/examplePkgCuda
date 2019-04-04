// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// [[Rcpp::export]]
arma::mat cublas_mat_mult(arma::mat const& A, arma::mat const& B) {
  cudaError_t cudaStat;    
  cublasStatus_t stat;
  cublasHandle_t handle;

  int m = A.n_rows;
  int n = A.n_cols;
  int k = B.n_cols;
  
  arma::mat C(m,k);
  double *dev_A, *dev_B, *dev_C;
  
  cudaMalloc((void**)&dev_A, m*n * sizeof(double));
  cudaMalloc((void**)&dev_B, n*k * sizeof(double));
  cudaMalloc((void**)&dev_C, m*k * sizeof(double));
  
  
  cublasCreate(&handle);
  
  cublasSetMatrix(m, n, sizeof(double), A.memptr(), m, dev_A, m);
  cublasSetMatrix(n, k, sizeof(double), B.memptr(), n, dev_B, k);
  
  double alpha = 1;
  double beta = 0;
  
  cublasDgemm(handle,
              CUBLAS_OP_N, CUBLAS_OP_N,
              m, n, k,
              &alpha,
              dev_A, m,
              dev_B, n,
              &beta,
              dev_C, m);
  
  cublasGetMatrix(m, k, sizeof(double), dev_C, m, C.memptr(), m);
  
  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_C);
  
  cublasDestroy(handle);
  
  return C;
}