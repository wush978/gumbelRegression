library(methods)
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

fold.id <- get.fold.id(n, fold.size <- 3)
lambda.seq <- c(1, .5, .25)
result <- gumbelRegression::gumbelRegression(X, y, fold.id, lambda.seq = lambda.seq)
library(Matrix)
options(gumbelRegression.parallel = FALSE, gumbelRegression.verbose = 2L)
result.cpp <- gumbelRegression::gumbelRegression(as(X, "CsparseMatrix"), y, fold.id, lambda.seq = lambda.seq, implementation = "cpp")

for(target in 1:fold.size) {
  for(i in seq_along(lambda.seq)) {
    w <- result.cpp[[target]]$coef[,i]
    mu <- X[fold.id == target,] %*% tail(w, -1)
    pred <- mu - digamma(1) * exp(w[1])
    e <- (y[fold.id == target] - pred)
    stopifnot(isTRUE(all.equal(result.cpp[[target]]$cv.mse[i], sum(e^2))))

    w <- result[[target]]$coef[,i]
    mu <- X[fold.id == target,] %*% tail(w, -1)
    pred <- mu - digamma(1) * exp(w[1])
    e <- (y[fold.id == target] - pred)
    stopifnot(isTRUE(all.equal(result[[target]]$cv.mse[i], sum(e^2))))
    print(abs(1 - result.cpp[[target]]$cv.mse[i] / result[[target]]$cv.mse[i]))
    stopifnot(abs(1 - result.cpp[[target]]$cv.mse[i] / result[[target]]$cv.mse[i]) < 1e-2)
  }
}

options(gumbelRegression.parallel = TRUE, gumbelRegression.verbose = 1L)
result.cpp <- gumbelRegression::gumbelRegression(as(X, "CsparseMatrix"), y, fold.id, lambda.seq = lambda.seq, implementation = "cpp")

for(target in 1:fold.size) {
  for(i in seq_along(lambda.seq)) {
    w <- result.cpp[[target]]$coef[,i]
    mu <- X[fold.id == target,] %*% tail(w, -1)
    pred <- mu - digamma(1) * exp(w[1])
    e <- (y[fold.id == target] - pred)
    stopifnot(isTRUE(all.equal(result.cpp[[target]]$cv.mse[i], sum(e^2))))

    w <- result[[target]]$coef[,i]
    mu <- X[fold.id == target,] %*% tail(w, -1)
    pred <- mu - digamma(1) * exp(w[1])
    e <- (y[fold.id == target] - pred)
    stopifnot(isTRUE(all.equal(result[[target]]$cv.mse[i], sum(e^2))))

    stopifnot(abs(1 - result.cpp[[target]]$cv.mse[i] / result[[target]]$cv.mse[i]) < 1e-2)
  }
}

loss <- get.loss(X, y)
for(i in 1:4) {
  cat(sprintf("%f -- %f\n", loss(result[[i]]$coef[,length(lambda.seq)]), loss(result.cpp[[i]]$coef[,length(lambda.seq)])))
}



stopifnot(all.equal(
  lapply(result, names),
  lapply(result.cpp, names)
))

stopifnot(all.equal(
  lapply(result, lapply, dim),
  lapply(result.cpp, lapply, dim)
))


