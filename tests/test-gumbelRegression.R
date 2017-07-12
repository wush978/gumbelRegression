library(methods)
set.seed(1)
library(gumbelRegression)
n <- 10
p <- 10
X <- cbind(1, matrix(rnorm(n * p), ncol = p))
alpha <- rnorm(p + 1)
beta <- rnorm(p + 1)
a <- X %*% alpha
b <- exp(X %*% beta)
y <- reliaR::rgumbel(n, a, 1/b)
# stopifnot(isTRUE(all.equal(reliaR::dgumbel(y, a, 1/b), QRM::dGumbel(y, a, 1/b))))
options("gumbelRegression.implementation" = "r")
if (isNamespaceLoaded("Xv")) unloadNamespace("Xv")
library(Matrix)
test <- function(X, y) {
  X.dgC <- as(X, "CsparseMatrix")
  loss <- get.loss(X, y, "r")
  loss.cpp <- get.loss(X.dgC, y, "cpp")
  gradient <- get.gradient(X, y)
  gradient.cpp <- get.gradient(X.dgC, y, "cpp")
  Hv <- get.Hv(X, y)
  Hv.cpp <- get.Hv(X.dgC, y, "cpp")

  w <- c(log(mean(1 / b)), alpha + rnorm(ncol(X), 0, 0.1))
  mu <- X %*% tail(w, -1)
  if (isS4(mu)) mu <- mu@x
  .w0 <- w;.w0[1] <- .w0[1] + 1
  .w1 <- w;.w1[-1] <- .w1[-1] + 1
  .w2 <- w + 1
  stopifnot(isTRUE(all.equal(loss(w), - sum(reliaR::dgumbel(y, mu, exp(w[1]), log = TRUE)))))
  stopifnot(isTRUE(all.equal(loss(w), loss.cpp(w))))
  stopifnot(isTRUE(all.equal(
    loss(w),
    gumbelRegression:::.test.loss.cpp(X.dgC, y, w, w[1], 0)
  )))
  stopifnot(isTRUE(all.equal(
    loss(w),
    gumbelRegression:::.test.loss.cpp(X.dgC, y, w, tail(w, -1), 1)
  )))
  stopifnot(isTRUE(all.equal(
    loss(w),
    gumbelRegression:::.test.loss.cpp(X.dgC, y, w, w, 2)
  )))
  stopifnot(isTRUE(all.equal(
    loss(.w0),
    gumbelRegression:::.test.loss.cpp(X.dgC, y, w, .w2[1], 0)
  )))
  stopifnot(isTRUE(all.equal(
    loss(.w1),
    gumbelRegression:::.test.loss.cpp(X.dgC, y, w, tail(.w2, -1), 1)
  )))
  stopifnot(isTRUE(all.equal(
    loss(.w2),
    gumbelRegression:::.test.loss.cpp(X.dgC, y, w, .w2, 2)
  )))
  stopifnot(isTRUE(all.equal(gradient(w), numDeriv::grad(loss, w))))
  stopifnot(isTRUE(all.equal(gradient(w), gradient.cpp(w))))
  stopifnot(isTRUE(all.equal(
    gradient(w)[1],
    gumbelRegression:::.test.gradient.cpp(X.dgC, y, w, w[1], 0)
  )))
  stopifnot(isTRUE(all.equal(
    tail(gradient(w), -1),
    gumbelRegression:::.test.gradient.cpp(X.dgC, y, w, tail(w, -1), 1)
  )))
  stopifnot(isTRUE(all.equal(
    gradient(w),
    gumbelRegression:::.test.gradient.cpp(X.dgC, y, w, w, 2)
  )))
  stopifnot(isTRUE(all.equal(
    gradient(.w0)[1],
    gumbelRegression:::.test.gradient.cpp(X.dgC, y, w, .w2[1], 0)
  )))
  stopifnot(isTRUE(all.equal(
    tail(gradient(.w1), -1),
    gumbelRegression:::.test.gradient.cpp(X.dgC, y, w, tail(.w2, -1), 1)
  )))
  stopifnot(isTRUE(all.equal(
    gradient(.w2),
    gumbelRegression:::.test.gradient.cpp(X.dgC, y, w, .w2, 2)
  )))
  H <- numDeriv::hessian(loss, w)
  H0 <- numDeriv::hessian(loss, .w0)
  H1 <- numDeriv::hessian(loss, .w1)
  H2 <- numDeriv::hessian(loss, .w2)
  for(i in seq_len(ncol(X) + 1)) {
    e <- rep(0, ncol(X) + 1)
    e[i] <- 1
    if (!isTRUE(all.equal(Hv(w, e), as.vector(H %*% e),tolerance = 1e-7 ))) {
      stop("Hv is inconsistent with numDeriv")
    }
    if (!isTRUE(all.equal(Hv(w, e), Hv.cpp(w, e), tolerance = 1e-7))) {
      stop("Hv is inconsistent with Hv.cpp")
    }
    stopifnot(isTRUE(all.equal(
      H[1,1] * e[1],
      gumbelRegression:::.test.Hv.cpp(X.dgC, y, w, w[1], e[1], 0)
    )))
    stopifnot(isTRUE(all.equal(tolerance = 1e-7, 
      as.vector(H[-1,-1] %*% e[-1]),
      gumbelRegression:::.test.Hv.cpp(X.dgC, y, w, w[-1], e[-1], 1)
    )))
    stopifnot(isTRUE(all.equal(tolerance = 1e-7, 
      as.vector(H %*% e),
      gumbelRegression:::.test.Hv.cpp(X.dgC, y, w, w, e, 2)
    )))
    stopifnot(isTRUE(all.equal(
      H0[1,1] * e[1],
      gumbelRegression:::.test.Hv.cpp(X.dgC, y, w, .w0[1], e[1], 0)
    )))
    stopifnot(isTRUE(all.equal(tolerance = 1e-7, 
      as.vector(H1[-1,-1] %*% e[-1]),
      gumbelRegression:::.test.Hv.cpp(X.dgC, y, w, .w1[-1], e[-1], 1)
    )))
    stopifnot(isTRUE(all.equal(tolerance = 1e-6, 
      as.vector(H2 %*% e),
      gumbelRegression:::.test.Hv.cpp(X.dgC, y, w, .w2, e, 2)
    )))
  }
}
test(X, y)
test(methods::as(X, "CsparseMatrix"), y)
test(methods::as(X, "RsparseMatrix"), y)
test(methods::as(X, "TsparseMatrix"), y)
library(Xv)
test(methods::as(X, "CsparseMatrix"), y)
test(methods::as(X, "RsparseMatrix"), y)
test(methods::as(X, "TsparseMatrix"), y)
