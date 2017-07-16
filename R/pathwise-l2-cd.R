#'@export
#'@title Generate \code{fold.id} of Given Size and Length
#'@param n integer value. The length of \code{fold.id}
#'@param nfold integer value. The number of folds.
#'@return integer vector. The value should between 1 and nfold.
get.fold.id <- function(n, nfold) {
  result <- rep(seq_len(nfold), ceiling(n / nfold))
  result <- head(result, n)
  sample(result)
}

#'@export
#'@title Pathwise L2 Regularization and Coordinate Descent
#'@param X matrix. The covariates.
#'@param y numeric vector. The response.
#'@param fold.id integer vector. The fold id of each instances. The number should be 1 to \code{nfold}.
#'@param lambda.seq decreasing numeric vector. The sequence of L2 regularization.
#'@param implementation character value. Select the implemenation.
#'@importFrom magrittr %>%
gumbelRegression <- function(X, y, fold.id, lambda.seq = 10^seq(1, -4, length.out = 100), tolerance = 1e-4, implementation = getOption("gumbelRegression.implementation", "r")) {
  switch(
    implementation,
    "r" = .gumbelRegression.r(X, y, fold.id, lambda.seq, tolerance),
    "cpp" = .gumbelRegression.cpp(X, y, fold.id, lambda.seq, tolerance),
    stop("Unknown implementation")
  )
}

.mse <- function(y, pred) sum((y - pred)^2)

.gumbelRegression.r <- function(X, y, fold.id, lambda.seq, tolerance) {
  library(HsTrust)
  get.f <- function(X, y, l2) {
    force(X)
    force(y)
    f0 <- get.loss(X, y)
    function(w) {
      f0(w) + l2 * sum(tail(w, -2)^2)
    }
  }

  get.g <- function(X, y, l2) {
    force(X)
    force(y)
    g0 <- get.gradient(X, y)
    function(w) {
      g0(w) + (2 * l2) * c(0, 0, tail(w, -2))
    }
  }

  get.Hv <- function(X, y, l2) {
    force(X)
    force(y)
    Hv0 <- gumbelRegression::get.Hv(X, y)
    function(w, v) {
      Hv0(w, v) + (2 * l2) * c(0, 0, tail(v, -2))
    }
  }

  get.projection <- function(f, w0, i) {
    force(f)
    force(w0)
    force(i)
    function(w) {
      .w <- w0
      .w[i] <- w
      result <- f(.w)
      if (length(result) > 1) result[i] else result
    }
  }

  get.projection.Hv <- function(f, w0, i) {
    force(f)
    force(w0)
    force(i)
    function(w, v) {
      .w <- w0
      .w[i] <- w
      .v <- rep(0, length(w0))
      .v[i] <- v
      result <- f(.w, .v)
      if (length(result) > 1) result[i] else result
    }
  }

  nfold <- max(fold.id)
  stopifnot(min(fold.id) == 1)
  stopifnot(nrow(X) == length(fold.id))
  stopifnot(nrow(X) == length(y))
  stopifnot(diff(lambda.seq) < 0)
  .moment <- get.moment(y)
  .fold.task <- function(fold.target) {
    if (fold.target <= nfold) {
      cv.train <- list(
        X = X[fold.id != fold.target,],
        y = y[fold.id != fold.target]
      )
      cv.test <- list(
        X = X[fold.id == fold.target,],
        y = y[fold.id == fold.target]
      )
      .start <- start <- c(log(.moment$sigma), .moment$mu, rep(0, ncol(cv.train$X) - 1))
      gumbel.coef <- matrix(.0, length(start), length(lambda.seq))
      cv.mse <- numeric(length(lambda.seq))
      for(i in seq_along(lambda.seq)) {
        l2 <- lambda.seq[i]
        repeat {
          old.start <- start
          start[1] <- optim(
            start[1],
            fn = get.f(cv.train$X, cv.train$y, l2) %>% get.projection(start, 1),
            gr = get.g(cv.train$X, cv.train$y, l2) %>% get.projection(start, 1),
            method = "BFGS",
            control = list(trace = 0)
          )$par
          kernel <- new(
            HsTrust::HsTrust,
            get.f(cv.train$X, cv.train$y, l2) %>% get.projection(start, seq(from = 2, by = 1, length.out = ncol(cv.train$X))),
            get.g(cv.train$X, cv.train$y, l2) %>% get.projection(start, seq(from = 2, by = 1, length.out = ncol(cv.train$X))),
            get.Hv(cv.train$X, cv.train$y, l2) %>% get.projection.Hv(start, seq(from = 2, by = 1, length.out = ncol(cv.train$X))),
            ncol(cv.train$X)
          )
          start[-1] <- kernel$tron_with_begin(tolerance, FALSE, tail(start, -1))
          if (abs(old.start - start) %>% max() < 1e-4) break
        }
        cv.mse[i] <- .mse(cv.test$y, cv.test$X %*% tail(start, -1) - digamma(1) * exp(start[1]))
        gumbel.coef[,i] <- start
        sprintf("fold: %d l2: %f mse: %f\n", fold.target, l2, cv.mse[i]) %>% cat()
      } # for i
      list(coef = gumbel.coef, cv.mse = cv.mse)
    } else { # if fold.target == nfold
      cv.train <- list(
        X = X,
        y = y
      )
      start <- gumbelRegression::get.moment(cv.train$y)
      .start <- start <- c(log(start$sigma), start$mu, rep(0, ncol(cv.train$X) - 1))
      gumbel.coef <- matrix(.0, length(start), length(lambda.seq))
      for(i in seq_along(lambda.seq)) {
        l2 <- lambda.seq[i]
        repeat {
          old.start <- start
          start[1] <- optim(
            start[1],
            fn = get.f(cv.train$X, cv.train$y, l2) %>% get.projection(start, 1),
            gr = get.g(cv.train$X, cv.train$y, l2) %>% get.projection(start, 1),
            method = "BFGS",
            control = list(trace = 0)
          )$par
          kernel <- new(
            HsTrust,
            get.f(cv.train$X, cv.train$y, l2) %>% get.projection(start, seq(from = 2, by = 1, length.out = ncol(cv.train$X))),
            get.g(cv.train$X, cv.train$y, l2) %>% get.projection(start, seq(from = 2, by = 1, length.out = ncol(cv.train$X))),
            get.Hv(cv.train$X, cv.train$y, l2) %>% get.projection.Hv(start, seq(from = 2, by = 1, length.out = ncol(cv.train$X))),
            ncol(cv.train$X)
          )
          start[-1] <- kernel$tron_with_begin(tolerance, FALSE, tail(start, -1))
          if (abs(old.start - start) %>% max() < tolerance) break
        }
        gumbel.coef[,i] <- start
      }
      list(coef = gumbel.coef)
    }
  }

  lapply(seq_len(nfold + 1), .fold.task)
}

.gumbelRegression.cpp <- function(X, y, fold.id, lambda.seq, tolerance, parallel) {
  stopifnot(isS4(X))
  stopifnot(class(X) == "dgCMatrix")
  nfold <- max(fold.id)
  stopifnot(min(fold.id) == 1)
  stopifnot(nrow(X) == length(fold.id))
  stopifnot(nrow(X) == length(y))
  stopifnot(diff(lambda.seq) < 0)
  .moment <- get.moment(y)
  .gumbelRegression.cpp.internal(X, y, fold.id, lambda.seq, tolerance, log(.moment$sigma), .moment$mu, verbose = getOption("gumbelRegression.verbose", 0L), parallel = getOption("gumbelRegression.parallel", TRUE))
}
