.get.loss.r <- function(X, y) {
  function(w) {
    sigma <- exp(w[1])
    w1 <- tail(w, -1)
    mu <- X %*% w1
    if (isS4(mu)) mu <- mu@x
    z <- (y - mu) / sigma
    sum(z + exp(-z) + log(sigma))
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
get.loss <- function(X, y, implementation = c("r", "cpp")) {
  switch(
    implementation[1],
    "r" = .get.loss.r(X, y),
    "cpp" = .get.loss.cpp(X, y),
    stop("Unknown implementation")
    )
}

.get.gradient.r <- function(X, y) {
  n <- nrow(X)
  function(w) {
    sigma <- exp(w[1])
    w1 <- tail(w, -1)
    mu <- (X %*% w1)
    if (isS4(mu)) mu <- mu@x
    z <- (y - mu) / sigma
    .z <- (as.vector(exp(-z) - 1) %*% X)
    if (isS4(.z)) .z <- .z@x
    c(
      n + sum(z * (exp(-z) - 1)),
      .z / sigma
    )
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
get.gradient <- function(X, y, implementation = c("r", "cpp")) {
  switch(
    implementation[1],
    "r" = .get.gradient.r(X, y),
    "cpp" = .get.gradient.cpp(X, y),
    stop("Unknown implementation")
    )
}

.get.Hv.r <- function(X, y) {

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
get.Hv <- function(X, y, implementation = c("r", "cpp")) {
  switch(
    implementation[1],
    "r" = .get.Hv.r(X, y),
    "cpp" = .get.Hv.cpp(X, y),
    stop("Unknown implementation")
    )
}
