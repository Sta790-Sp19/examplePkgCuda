// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

// [[Rcpp::export]]
arma::mat cusolver_chol(arma::mat const& A) {
  int n = A.n_rows;;
  
  arma::mat L(n, n, arma::fill::zeros);
  double *dev_A;
  
  cudaMalloc((void**)&dev_A, n*n * sizeof(double));
  
  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);
  
  cublasSetMatrix(n, n, sizeof(double), A.memptr(), n, dev_A, n);
  
  int work_size = 0;
  cusolverDnDpotrf_bufferSize(
    handle, CUBLAS_FILL_MODE_LOWER, 
    n, dev_A, n, 
    &work_size
  );
  
  double *work;
  cudaMalloc(&work, work_size * sizeof(double));
  
  int *dev_info;
  cudaMalloc((void **)&dev_info, sizeof(int));
  
  cusolverDnDpotrf(
    handle, CUBLAS_FILL_MODE_LOWER, 
    n, dev_A, n, 
    work, work_size, 
    dev_info
  );
  
  int info = 0;
  cudaMemcpy(&info, dev_info, sizeof(int), cudaMemcpyDeviceToHost);
  
  Rcpp::Rcout << "work_size: " << work_size << "\n";
  Rcpp::Rcout << "dev_info: " << info << "\n";
  
  cublasGetMatrix(n, n, sizeof(double), dev_A, n, L.memptr(), n);
  //cudaMemcpy(L.memptr(), dev_A, n * n * sizeof(double), cudaMemcpyDeviceToHost);
  
  cudaFree(dev_A);
  cudaFree(work);
  cudaFree(dev_info);
  
  cusolverDnDestroy(handle);
  
  return L;
}