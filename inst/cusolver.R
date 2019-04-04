library(dplyr)

d = tibble(
  x = runif(10000),
  y = runif(10000)
) %>%
  dist() %>%
  as.matrix()

 
Sigma = 1 * exp(-(d^2)) + diag(0.8, 10000,10000)

system.time(chol(Sigma))
system.time(cusolver_chol(Sigma))
