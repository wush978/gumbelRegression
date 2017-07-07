#'@title Moment Estimator of Gumbel Distribution
#'@param y numeric vector.
#'@return A list with following elements:
#'\itemize{
#'  \item mu. The location parameter.
#'  \item sigma. The scale parameter.
#'}
#'@export
get.moment <- function(y) {
  sigma <- sqrt(6 * var(y)) / pi
  mu <- mean(y) + digamma(1) * sigma
  list(mu = mu, sigma = sigma)
}

.extract.w <- function(X, y, w) {
  sigma <- exp(w[1])
  w1 <- tail(w, -1)
  mu <- (X %*% w1)
  if (isS4(mu)) mu <- mu@x
  mu <- as.vector(mu)
  z <- as.vector((y - mu) / sigma)
  list(sigma = sigma, mu = mu, z = z)
}

.threshold.positive <- .Machine$double.xmax / 10
.threshold.negative <- - .Machine$double.xmax / 10
.threshold.correction <- function(x) {
  x[x > .threshold.positive] <- .threshold.positive
  x[x < .threshold.negative] <- .threshold.negative
  x
}

.get.loss.r <- function(X, y) {
  force(X)
  force(y)
  function(w) {
    .w <- .extract.w(X, y, w)
    result <- sum(.w$z + exp(-.w$z) + log(.w$sigma))
    .threshold.correction(result)
  }
}

#'@title Get the Loss Function of Gumbel Regression Given the Response and Covariates
#'@param X matrix. The covariates.
#'@param y numeric vector. The response.
#'@param implementation character value. Select the implemenation.
#'@return function with single parameter which should be a numeric vector \code{w}.
#'The first element of \code{w} is the log of scale parameter of gumbel distribution.
#'The other elements are regression coefficients.
#'@export
get.loss <- function(X, y, implementation = getOption("gumbelRegression.implementation", "r")) {
  switch(
    implementation[1],
    "r" = .get.loss.r(X, y),
    "cpp" = .get.loss.cpp(X, y),
    stop("Unknown implementation")
    )
}

.get.gradient.r <- function(X, y) {
  force(X)
  force(y)
  n <- nrow(X)
  function(w) {
    .w <- .extract.w(X, y, w)
    .z <- (as.vector(exp(-.w$z) - 1) %*% X)
    if (isS4(.z)) .z <- .z@x
    result <- c(
      n + sum(.w$z * (exp(-.w$z) - 1)),
      .z / .w$sigma
    )
    .threshold.correction(result)
  }
}

#'@title Get the Gradient of the Loss Function of Gumbel Regression Given the Response and Covariates
#'@param X matrix. The covariates.
#'@param y numeric vector. The response.
#'@param implementation character value. Select the implemenation.
#'@return function with single parameter which should be a numeric vector \code{w}.
#'The first element of \code{w} is the log of scale parameter of gumbel distribution.
#'The other elements are regression coefficients.
#'@export
get.gradient <- function(X, y, implementation = getOption("gumbelRegression.implementation", "r")) {
  switch(
    implementation[1],
    "r" = .get.gradient.r(X, y),
    "cpp" = .get.gradient.cpp(X, y),
    stop("Unknown implementation")
    )
}

.get.Hv.r <- function(X, y) {
  force(X)
  force(y)
  function(w, v) {
    .w <- .extract.w(X, y, w)
    .e.z <- exp(-.w$z)
    H11 <- sum((.w$z^2 - .w$z) * .e.z + .w$z)
    H1n <- (((.w$z - 1) * .e.z + 1) / .w$sigma) %*% X
    Hnn <- .e.z / .w$sigma^2
    if (isS4(H1n)) H1n <- H1n@x
    v1 <- tail(v, -1)
    Xv1 <- X %*% v1
    if (isS4(Xv1)) Xv1 <- Xv1@x
    XXv1 <- (as.vector(Xv1) * Hnn) %*% X
    if (isS4(XXv1)) XXv1 <- XXv1@x
    result <- c(
      H11 * v[1] + sum(H1n * v1),
      H1n * v[1] + XXv1
    )
    .threshold.correction(result)
  }
}

#'@title Get the Hessian-Vector Multiplication Function of Gumbel Regression Given the Response and Covariates
#'@param X matrix. The covariates.
#'@param y numeric vector. The response.
#'@param implementation character value. Select the implemenation.
#'@return function with two parameters: \code{w} and \code{s}.
#'The first element of \code{w} is the log of scale parameter of gumbel distribution.
#'The other elements are regression coefficients.
#'The \code{s} is the vector which will be multiplied by the hessian.
#'@export
get.Hv <- function(X, y, implementation = getOption("gumbelRegression.implementation", "r")) {
  switch(
    implementation[1],
    "r" = .get.Hv.r(X, y),
    "cpp" = .get.Hv.cpp(X, y),
    stop("Unknown implementation")
    )
}
