library(methods)
library(gumbelRegression)
library(Matrix)
library(HsTrust)
set.seed(1)
n <- 10000
p <- 100
X <- cbind(1, matrix(sample(0:1, n * p, TRUE, c(0.9, 0.1)), ncol = p))
alpha <- rnorm(p + 1)
beta <- rnorm(p + 1)
a <- X %*% alpha
# b <- exp(X %*% beta)
b <- rexp(1)
y <- reliaR::rgumbel(100, a, 1/b)
X <- as(X, "CsparseMatrix")

fold.id <- get.fold.id(n, fold.size <- 2)
lambda.seq <- seq(1, -4, length.out = 100)
options(gumbelRegression.parallel = TRUE, gumbelRegression.verbose = 3L)
result.cpp <- gumbelRegression::gumbelRegression(X, y, fold.id, lambda.seq = (10^lambda.seq), implementation = "cpp")
