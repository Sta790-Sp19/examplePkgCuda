// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// cublas_mat_mult
arma::mat cublas_mat_mult(arma::mat const& A, arma::mat const& B);
RcppExport SEXP _examplePkgCuda_cublas_mat_mult(SEXP ASEXP, SEXP BSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat const& >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type B(BSEXP);
    rcpp_result_gen = Rcpp::wrap(cublas_mat_mult(A, B));
    return rcpp_result_gen;
END_RCPP
}
// cublas_mat_mult2
arma::mat cublas_mat_mult2(arma::mat const& A, arma::mat const& B);
RcppExport SEXP _examplePkgCuda_cublas_mat_mult2(SEXP ASEXP, SEXP BSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat const& >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat const& >::type B(BSEXP);
    rcpp_result_gen = Rcpp::wrap(cublas_mat_mult2(A, B));
    return rcpp_result_gen;
END_RCPP
}
// curand_rnorm
std::vector<double> curand_rnorm(int n, int seed, double mu, double sigma);
RcppExport SEXP _examplePkgCuda_curand_rnorm(SEXP nSEXP, SEXP seedSEXP, SEXP muSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< double >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(curand_rnorm(n, seed, mu, sigma));
    return rcpp_result_gen;
END_RCPP
}
// cusolver_chol
arma::mat cusolver_chol(arma::mat const& A);
RcppExport SEXP _examplePkgCuda_cusolver_chol(SEXP ASEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat const& >::type A(ASEXP);
    rcpp_result_gen = Rcpp::wrap(cusolver_chol(A));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_hello_world
List rcpp_hello_world();
RcppExport SEXP _examplePkgCuda_rcpp_hello_world() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(rcpp_hello_world());
    return rcpp_result_gen;
END_RCPP
}
// rcpp_sum
double rcpp_sum(Rcpp::NumericVector v);
RcppExport SEXP _examplePkgCuda_rcpp_sum(SEXP vSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type v(vSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_sum(v));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_thrust_version
void rcpp_thrust_version();
RcppExport SEXP _examplePkgCuda_rcpp_thrust_version() {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_thrust_version();
    return R_NilValue;
END_RCPP
}
// rcpp_gpu_sq_exp_cov
Rcpp::NumericMatrix rcpp_gpu_sq_exp_cov(Rcpp::NumericMatrix const& d, double sigma2, double range);
RcppExport SEXP _examplePkgCuda_rcpp_gpu_sq_exp_cov(SEXP dSEXP, SEXP sigma2SEXP, SEXP rangeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix const& >::type d(dSEXP);
    Rcpp::traits::input_parameter< double >::type sigma2(sigma2SEXP);
    Rcpp::traits::input_parameter< double >::type range(rangeSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_gpu_sq_exp_cov(d, sigma2, range));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_examplePkgCuda_cublas_mat_mult", (DL_FUNC) &_examplePkgCuda_cublas_mat_mult, 2},
    {"_examplePkgCuda_cublas_mat_mult2", (DL_FUNC) &_examplePkgCuda_cublas_mat_mult2, 2},
    {"_examplePkgCuda_curand_rnorm", (DL_FUNC) &_examplePkgCuda_curand_rnorm, 4},
    {"_examplePkgCuda_cusolver_chol", (DL_FUNC) &_examplePkgCuda_cusolver_chol, 1},
    {"_examplePkgCuda_rcpp_hello_world", (DL_FUNC) &_examplePkgCuda_rcpp_hello_world, 0},
    {"_examplePkgCuda_rcpp_sum", (DL_FUNC) &_examplePkgCuda_rcpp_sum, 1},
    {"_examplePkgCuda_rcpp_thrust_version", (DL_FUNC) &_examplePkgCuda_rcpp_thrust_version, 0},
    {"_examplePkgCuda_rcpp_gpu_sq_exp_cov", (DL_FUNC) &_examplePkgCuda_rcpp_gpu_sq_exp_cov, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_examplePkgCuda(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
