n = 1e4

A = matrix(runif(n*n), n, n)
B = matrix(runif(n*n), n, n)

system.time({C = A %*% B})
system.time({C_gpu = cublas_mat_mult(A,B)})

identical(C, C_gpu)
diff = C - C_gpu
sum(diff)
