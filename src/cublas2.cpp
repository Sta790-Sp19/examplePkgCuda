// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

class cublas_handle {
  cublasHandle_t handle;

public:
  cublas_handle() {
    cublasStatus_t res = cublasCreate(&handle);
    if (res != CUBLAS_STATUS_SUCCESS) {
      Rcpp::stop("cuBLAS initialization failed.");
    }
  }
  
  ~cublas_handle() {
    cublasDestroy(handle);
  }
  
  cublasHandle_t get_handle() {
    return handle;
  }
};

class cuda_mat {
  double* d_ptr;
  
public:
  const int n_rows, n_cols;
  cuda_mat(arma::mat const& m)
    : n_rows(m.n_rows), n_cols(m.n_cols)
  {
    cudaMalloc(&d_ptr, n_rows*n_cols * sizeof(double));
    cublasSetMatrix(n_rows, n_cols, sizeof(double), m.memptr(), n_rows, d_ptr, n_rows);
  }
  
  cuda_mat(int m, int n)
    : n_rows(m), n_cols(n)
  {
    cudaMalloc(&d_ptr, m*n * sizeof(double));
  }
  
  ~cuda_mat() {
    cudaFree(d_ptr);
  }
  
  double* get_dev_ptr() {
    return d_ptr;
  }
  
  arma::mat get_mat() {
    arma::mat m(n_rows,n_cols);
    cublasGetMatrix(n_rows, n_cols, sizeof(double), d_ptr, n_rows, m.memptr(), n_rows);
    return m;
  }
};


// [[Rcpp::export]]
arma::mat cublas_mat_mult2(arma::mat const& A, arma::mat const& B) {
  
  cublas_handle h;
  
  int m = A.n_rows;
  int n = A.n_cols;
  int k = B.n_cols;
  
  cuda_mat cA(A);
  cuda_mat cB(B);
  cuda_mat cC(m,k);
    
  double alpha = 1;
  double beta = 0;
    
  cublasDgemm(h.get_handle(),
              CUBLAS_OP_N, CUBLAS_OP_N,
              m, n, k,
              &alpha,
              cA.get_dev_ptr(), m,
              cB.get_dev_ptr(), n,
              &beta,
              cC.get_dev_ptr(), m);  
  
  return cC.get_mat();
}