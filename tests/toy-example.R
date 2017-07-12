library(gumbelRegression)
set.seed(1)
n <- 100
p <- 5
X <- cbind(1, matrix(rnorm(n * p), ncol = p))
alpha <- rnorm(p + 1)
beta <- rnorm(p + 1)
a <- X %*% alpha
b <- exp(X %*% beta)
y <- reliaR::rgumbel(100, a, 1/b)

result <- gumbelRegression::gumbelRegression(X, y, get.fold.id(n, 3))
library(Matrix)
result.cpp <- gumbelRegression::gumbelRegression(as(X, "CsparseMatrix"), y, get.fold.id(n, 3), implementation = "cpp")

stopifnot(all.equal(
  lapply(result, names),
  lapply(result.cpp, names)
))

stopifnot(all.equal(
  lapply(result, lapply, dim),
  lapply(result.cpp, lapply, dim)
))
