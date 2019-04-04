// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>


// [[Rcpp::export]]
arma::mat cusolver_chol(arma::mat const& A) {
  
  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);
  
  int n = A.n_rows;
  
  double *dev_A;
  cudaMalloc(&dev_A, n*n * sizeof(double));
  cublasSetMatrix(n, n, sizeof(double), A.memptr(), n, dev_A, n);
  
  // Actual Decomposition
  
  int work = 0;
  
  cusolverDnDpotrf_bufferSize(handle,
                              CUBLAS_FILL_MODE_LOWER,
                              n,
                              dev_A,
                              n,
                              &work);
  
  Rcpp::Rcout << "work size: " << work << "\n";
  
  double *dev_work;
  cudaMalloc(&dev_work, work * sizeof(double));
   
  int* dev_info;
  cudaMalloc(&dev_info, sizeof(int));
  
  cusolverDnDpotrf(handle,
                   CUBLAS_FILL_MODE_LOWER,
                   n,
                   dev_A,
                   n,
                   dev_work,
                   work,
                   dev_info);
   
  arma::mat L(n,n);
  cublasGetMatrix(n, n, sizeof(double), dev_A, n, L.memptr(), n);
  
  cudaFree(dev_A);
  cudaFree(dev_info);
  cudaFree(dev_work);
  cusolverDnDestroy(handle);
  
  return L;
}