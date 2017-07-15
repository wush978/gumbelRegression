library(Matrix)
library(gumbelRegression)
library(methods)
set.seed(1)
library(gumbelRegression)
n <- 300
p <- 10
X <- cbind(1, matrix(sample(0:1, n * p, TRUE, prob = c(0.9, 0.1)), ncol = p))
alpha <- rnorm(p + 1)
beta <- rnorm(p + 1)
a <- X %*% alpha
b <- exp(X %*% beta)
y <- reliaR::rgumbel(n, a, 1/b)

X <- as(X, "CsparseMatrix")
l2 <- rexp(1)
fold.id <- rep(1:3, ceiling(nrow(X) / 3))
fold.id <- head(fold.id, nrow(X))
fold.id <- sample(fold.id)
fold.target <- 1L
X.cv <- X[fold.id != fold.target,]
y.cv <- y[fold.id != fold.target]

loss <- get.loss(X, y, "r")
loss.l2 <- function(w) {
  loss(w) + l2 * sum(tail(w, -2)^2)
}
loss.cv <- get.loss(X.cv, y.cv, "r")
loss.cv.l2 <- function(w) {
  loss.cv(w) + l2 * sum(tail(w, -2)^2)
}

w <- c(log(mean(1 / b)), alpha + rnorm(ncol(X), 0, 0.1))
mu <- X %*% tail(w, -1)

.w0 <- w;.w0[1] <- .w0[1] + 1
.w1 <- w;.w1[-1] <- .w1[-1] + 1
.w2 <- w + 1

stopifnot(isTRUE(all.equal(
  loss(w),
  gumbelRegression:::.test.loss.cpp(X, y, w, w[1], 0)
)))
stopifnot(isTRUE(all.equal(
  loss(w),
  gumbelRegression:::.test.loss.cpp(X, y, w, tail(w, -1), 1)
)))
stopifnot(isTRUE(all.equal(
  loss(w),
  gumbelRegression:::.test.loss.cpp(X, y, w, w, 2)
)))
stopifnot(isTRUE(all.equal(
  loss(.w0),
  gumbelRegression:::.test.loss.cpp(X, y, w, .w2[1], 0)
)))
stopifnot(isTRUE(all.equal(
  loss(.w1),
  gumbelRegression:::.test.loss.cpp(X, y, w, tail(.w2, -1), 1)
)))
stopifnot(isTRUE(all.equal(
  loss(.w2),
  gumbelRegression:::.test.loss.cpp(X, y, w, .w2, 2)
)))
# l2
stopifnot(isTRUE(all.equal(
  loss.l2(w),
  gumbelRegression:::.test.loss.cpp(X, y, w, w[1], 0, l2)
)))
stopifnot(isTRUE(all.equal(
  loss.l2(w),
  gumbelRegression:::.test.loss.cpp(X, y, w, tail(w, -1), 1, l2)
)))
stopifnot(isTRUE(all.equal(
  loss.l2(w),
  gumbelRegression:::.test.loss.cpp(X, y, w, w, 2, l2)
)))
stopifnot(isTRUE(all.equal(
  loss.l2(.w0),
  gumbelRegression:::.test.loss.cpp(X, y, w, .w2[1], 0, l2)
)))
stopifnot(isTRUE(all.equal(
  loss.l2(.w1),
  gumbelRegression:::.test.loss.cpp(X, y, w, tail(.w2, -1), 1, l2)
)))
stopifnot(isTRUE(all.equal(
  loss.l2(.w2),
  gumbelRegression:::.test.loss.cpp(X, y, w, .w2, 2, l2)
)))
# cv
stopifnot(isTRUE(all.equal(
  loss.cv(w),
  gumbelRegression:::.test.loss.cpp(X, y, w, w[1], 0, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  loss.cv(w),
  gumbelRegression:::.test.loss.cpp(X, y, w, tail(w, -1), 1, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  loss.cv(w),
  gumbelRegression:::.test.loss.cpp(X, y, w, w, 2, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  loss.cv(.w0),
  gumbelRegression:::.test.loss.cpp(X, y, w, .w2[1], 0, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  loss.cv(.w1),
  gumbelRegression:::.test.loss.cpp(X, y, w, tail(.w2, -1), 1, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  loss.cv(.w2),
  gumbelRegression:::.test.loss.cpp(X, y, w, .w2, 2, foldId = fold.id, foldTarget = fold.target)
)))
# cv + l2
stopifnot(isTRUE(all.equal(
  loss.cv.l2(w),
  gumbelRegression:::.test.loss.cpp(X, y, w, w[1], 0, l2 = l2, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  loss.cv.l2(w),
  gumbelRegression:::.test.loss.cpp(X, y, w, tail(w, -1), 1, l2 = l2, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  loss.cv.l2(w),
  gumbelRegression:::.test.loss.cpp(X, y, w, w, 2, l2 = l2, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  loss.cv.l2(.w0),
  gumbelRegression:::.test.loss.cpp(X, y, w, .w2[1], 0, l2 = l2, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  loss.cv.l2(.w1),
  gumbelRegression:::.test.loss.cpp(X, y, w, tail(.w2, -1), 1, l2 = l2, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  loss.cv.l2(.w2),
  gumbelRegression:::.test.loss.cpp(X, y, w, .w2, 2, l2 = l2, foldId = fold.id, foldTarget = fold.target)
)))

# gradient
stopifnot(isTRUE(all.equal(
  gradient(w)[1],
  gumbelRegression:::.test.gradient.cpp(X, y, w, w[1], 0)
)))
stopifnot(isTRUE(all.equal(
  tail(gradient(w), -1),
  gumbelRegression:::.test.gradient.cpp(X, y, w, tail(w, -1), 1)
)))
stopifnot(isTRUE(all.equal(
  gradient(w),
  gumbelRegression:::.test.gradient.cpp(X, y, w, w, 2)
)))
stopifnot(isTRUE(all.equal(
  gradient(.w0)[1],
  gumbelRegression:::.test.gradient.cpp(X, y, w, .w2[1], 0)
)))
stopifnot(isTRUE(all.equal(
  tail(gradient(.w1), -1),
  gumbelRegression:::.test.gradient.cpp(X, y, w, tail(.w2, -1), 1)
)))
stopifnot(isTRUE(all.equal(
  gradient(.w2),
  gumbelRegression:::.test.gradient.cpp(X, y, w, .w2, 2)
)))
# l2
gradient.l2 <- function(w) {
  gradient(w) + 2 * l2 * c(0, 0, tail(w, -2))
}
stopifnot(isTRUE(all.equal(
  gradient.l2(w)[1],
  gumbelRegression:::.test.gradient.cpp(X, y, w, w[1], 0, l2 = l2)
)))
stopifnot(isTRUE(all.equal(
  tail(gradient.l2(w), -1),
  gumbelRegression:::.test.gradient.cpp(X, y, w, tail(w, -1), 1, l2 = l2)
)))
stopifnot(isTRUE(all.equal(
  gradient.l2(w),
  gumbelRegression:::.test.gradient.cpp(X, y, w, w, 2, l2 = l2)
)))
stopifnot(isTRUE(all.equal(
  gradient.l2(.w0)[1],
  gumbelRegression:::.test.gradient.cpp(X, y, w, .w2[1], 0, l2 = l2)
)))
stopifnot(isTRUE(all.equal(
  tail(gradient.l2(.w1), -1),
  gumbelRegression:::.test.gradient.cpp(X, y, w, tail(.w2, -1), 1, l2 = l2)
)))
stopifnot(isTRUE(all.equal(
  gradient.l2(.w2),
  gumbelRegression:::.test.gradient.cpp(X, y, w, .w2, 2, l2 = l2)
)))
# cv
gradient.cv <- get.gradient(X.cv, y.cv, "r")
stopifnot(isTRUE(all.equal(
  gradient.cv(w)[1],
  gumbelRegression:::.test.gradient.cpp(X, y, w, w[1], 0, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  tail(gradient.cv(w), -1),
  gumbelRegression:::.test.gradient.cpp(X, y, w, tail(w, -1), 1, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  gradient.cv(w),
  gumbelRegression:::.test.gradient.cpp(X, y, w, w, 2, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  gradient.cv(.w0)[1],
  gumbelRegression:::.test.gradient.cpp(X, y, w, .w2[1], 0, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  tail(gradient.cv(.w1), -1),
  gumbelRegression:::.test.gradient.cpp(X, y, w, tail(.w2, -1), 1, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  gradient.cv(.w2),
  gumbelRegression:::.test.gradient.cpp(X, y, w, .w2, 2, foldId = fold.id, foldTarget = fold.target)
)))
# cv l2
gradient.cv.l2 <- function(w) {
  gradient.cv(w) + 2 * l2 * c(0, 0, tail(w, -2))
}
stopifnot(isTRUE(all.equal(
  gradient.cv.l2(w)[1],
  gumbelRegression:::.test.gradient.cpp(X, y, w, w[1], 0, l2 = l2, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  tail(gradient.cv.l2(w), -1),
  gumbelRegression:::.test.gradient.cpp(X, y, w, tail(w, -1), 1, l2 = l2, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  gradient.cv.l2(w),
  gumbelRegression:::.test.gradient.cpp(X, y, w, w, 2, l2 = l2, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  gradient.cv.l2(.w0)[1],
  gumbelRegression:::.test.gradient.cpp(X, y, w, .w2[1], 0, l2 = l2, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  tail(gradient.cv.l2(.w1), -1),
  gumbelRegression:::.test.gradient.cpp(X, y, w, tail(.w2, -1), 1, l2 = l2, foldId = fold.id, foldTarget = fold.target)
)))
stopifnot(isTRUE(all.equal(
  gradient.cv.l2(.w2),
  gumbelRegression:::.test.gradient.cpp(X, y, w, .w2, 2, l2 = l2, foldId = fold.id, foldTarget = fold.target)
)))


H <- numDeriv::hessian(loss, w)
H.l2 <- numDeriv::hessian(loss.l2, w)
H.cv <- numDeriv::hessian(loss.cv, w)
H.cv.l2 <- numDeriv::hessian(loss.cv.l2, w)
H0 <- numDeriv::hessian(loss, .w0)
H0.l2 <- numDeriv::hessian(loss.l2, .w0)
H0.cv <- numDeriv::hessian(loss.cv, .w0)
H0.cv.l2 <- numDeriv::hessian(loss.cv.l2, .w0)
H1 <- numDeriv::hessian(loss, .w1)
H1.l2 <- numDeriv::hessian(loss.l2, .w1)
H1.cv <- numDeriv::hessian(loss.cv, .w1)
H1.cv.l2 <- numDeriv::hessian(loss.cv.l2, .w1)
H2 <- numDeriv::hessian(loss, .w2)
H2.l2 <- numDeriv::hessian(loss.l2, .w2)
H2.cv <- numDeriv::hessian(loss.cv, .w2)
H2.cv.l2 <- numDeriv::hessian(loss.cv.l2, .w2)
for(i in seq_len(ncol(X) + 1)) {
  e <- rep(0, ncol(X) + 1)
  e[i] <- 1
  stopifnot(isTRUE(all.equal(
    H[1,1] * e[1],
    gumbelRegression:::.test.Hv.cpp(X, y, w, w[1], e[1], 0)
  )))
  stopifnot(isTRUE(all.equal(tolerance = 1e-7,
    as.vector(H[-1,-1] %*% e[-1]),
    gumbelRegression:::.test.Hv.cpp(X, y, w, w[-1], e[-1], 1)
  )))
  stopifnot(isTRUE(all.equal(tolerance = 1e-7,
    as.vector(H %*% e),
    gumbelRegression:::.test.Hv.cpp(X, y, w, w, e, 2)
  )))
  stopifnot(isTRUE(all.equal(
    H0[1,1] * e[1],
    gumbelRegression:::.test.Hv.cpp(X, y, w, .w0[1], e[1], 0)
  )))
  stopifnot(isTRUE(all.equal(tolerance = 1e-7,
    as.vector(H1[-1,-1] %*% e[-1]),
    gumbelRegression:::.test.Hv.cpp(X, y, w, .w1[-1], e[-1], 1)
  )))
  stopifnot(isTRUE(all.equal(tolerance = 1e-6,
    as.vector(H2 %*% e),
    gumbelRegression:::.test.Hv.cpp(X, y, w, .w2, e, 2)
  )))
  # l2
  stopifnot(isTRUE(all.equal(
    H.l2[1,1] * e[1],
    gumbelRegression:::.test.Hv.cpp(X, y, w, w[1], e[1], 0, l2 = l2)
  )))
  stopifnot(isTRUE(all.equal(tolerance = 1e-7,
    as.vector(H.l2[-1,-1] %*% e[-1]),
    gumbelRegression:::.test.Hv.cpp(X, y, w, w[-1], e[-1], 1, l2 = l2)
  )))
  stopifnot(isTRUE(all.equal(tolerance = 1e-7,
    as.vector(H.l2 %*% e),
    gumbelRegression:::.test.Hv.cpp(X, y, w, w, e, 2, l2 = l2)
  )))
  stopifnot(isTRUE(all.equal(
    H0.l2[1,1] * e[1],
    gumbelRegression:::.test.Hv.cpp(X, y, w, .w0[1], e[1], 0, l2 = l2)
  )))
  stopifnot(isTRUE(all.equal(tolerance = 1e-7,
    as.vector(H1.l2[-1,-1] %*% e[-1]),
    gumbelRegression:::.test.Hv.cpp(X, y, w, .w1[-1], e[-1], 1, l2 = l2)
  )))
  stopifnot(isTRUE(all.equal(tolerance = 1e-6,
    as.vector(H2.l2 %*% e),
    gumbelRegression:::.test.Hv.cpp(X, y, w, .w2, e, 2, l2 = l2)
  )))
  # cv
  stopifnot(isTRUE(all.equal(
    H.cv[1,1] * e[1],
    gumbelRegression:::.test.Hv.cpp(X, y, w, w[1], e[1], 0, foldId = fold.id, foldTarget = fold.target)
  )))
  stopifnot(isTRUE(all.equal(tolerance = 1e-7,
    as.vector(H.cv[-1,-1] %*% e[-1]),
    gumbelRegression:::.test.Hv.cpp(X, y, w, w[-1], e[-1], 1, foldId = fold.id, foldTarget = fold.target)
  )))
  stopifnot(isTRUE(all.equal(tolerance = 1e-7,
    as.vector(H.cv %*% e),
    gumbelRegression:::.test.Hv.cpp(X, y, w, w, e, 2, foldId = fold.id, foldTarget = fold.target)
  )))
  stopifnot(isTRUE(all.equal(
    H0.cv[1,1] * e[1],
    gumbelRegression:::.test.Hv.cpp(X, y, w, .w0[1], e[1], 0, foldId = fold.id, foldTarget = fold.target)
  )))
  stopifnot(isTRUE(all.equal(tolerance = 1e-7,
    as.vector(H1.cv[-1,-1] %*% e[-1]),
    gumbelRegression:::.test.Hv.cpp(X, y, w, .w1[-1], e[-1], 1, foldId = fold.id, foldTarget = fold.target)
  )))
  stopifnot(isTRUE(all.equal(tolerance = 1e-6,
    as.vector(H2.cv %*% e),
    gumbelRegression:::.test.Hv.cpp(X, y, w, .w2, e, 2, foldId = fold.id, foldTarget = fold.target)
  )))
  # cv + l2
  stopifnot(isTRUE(all.equal(
    H.cv.l2[1,1] * e[1],
    gumbelRegression:::.test.Hv.cpp(X, y, w, w[1], e[1], 0, foldId = fold.id, foldTarget = fold.target, l2 = l2)
  )))
  stopifnot(isTRUE(all.equal(tolerance = 1e-7,
    as.vector(H.cv.l2[-1,-1] %*% e[-1]),
    gumbelRegression:::.test.Hv.cpp(X, y, w, w[-1], e[-1], 1, foldId = fold.id, foldTarget = fold.target, l2 = l2)
  )))
  stopifnot(isTRUE(all.equal(tolerance = 1e-7,
    as.vector(H.cv.l2 %*% e),
    gumbelRegression:::.test.Hv.cpp(X, y, w, w, e, 2, foldId = fold.id, foldTarget = fold.target, l2 = l2)
  )))
  stopifnot(isTRUE(all.equal(
    H0.cv.l2[1,1] * e[1],
    gumbelRegression:::.test.Hv.cpp(X, y, w, .w0[1], e[1], 0, foldId = fold.id, foldTarget = fold.target, l2 = l2)
  )))
  stopifnot(isTRUE(all.equal(tolerance = 1e-7,
    as.vector(H1.cv.l2[-1,-1] %*% e[-1]),
    gumbelRegression:::.test.Hv.cpp(X, y, w, .w1[-1], e[-1], 1, foldId = fold.id, foldTarget = fold.target, l2 = l2)
  )))
  stopifnot(isTRUE(all.equal(tolerance = 1e-6,
    as.vector(H2.cv.l2 %*% e),
    gumbelRegression:::.test.Hv.cpp(X, y, w, .w2, e, 2, foldId = fold.id, foldTarget = fold.target, l2 = l2)
  )))
}

