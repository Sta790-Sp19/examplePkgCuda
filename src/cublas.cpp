// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


// [[Rcpp::export]]
arma::mat cublas_mat_mult(arma::mat const& A, arma::mat const& B) {
  
  cublasHandle_t handle;
  cublasCreate(&handle);
  
  int m = A.n_rows;
  int n = A.n_cols;
  int k = B.n_cols;

  double *dev_A, *dev_B, *dev_C;
  
  cudaMalloc(&dev_A, m*n * sizeof(double));
  cudaMalloc(&dev_B, n*k * sizeof(double));
  cudaMalloc(&dev_C, m*k * sizeof(double));
    
  cublasSetMatrix(m, n, sizeof(double), A.memptr(), m, dev_A, m);
  cublasSetMatrix(n, k, sizeof(double), B.memptr(), n, dev_B, n);
    
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
  
  arma::mat C(m,k);
  cublasGetMatrix(m, k, sizeof(double), dev_C, m, C.memptr(), m);
  
  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_C);
  
  cublasDestroy(handle);
      
  return C;
}