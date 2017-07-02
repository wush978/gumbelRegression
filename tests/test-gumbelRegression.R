library(gumbelRegression)
n <- 10^2
p <- 10
X <- cbind(1, matrix(rnorm(n * p), ncol = p))
alpha <- rnorm(p + 1)
beta <- rnorm(p + 1)
a <- X %*% alpha
b <- exp(X %*% beta)
y <- reliaR::rgumbel(100, a, 1/b)
# stopifnot(isTRUE(all.equal(reliaR::dgumbel(y, a, 1/b), QRM::dGumbel(y, a, 1/b))))
loss <- get.loss(X, y)
gradient <- get.gradient(X, y)

w <- c(log(mean(1 / b)), alpha + rnorm(ncol(X), 0, 0.1))
stopifnot(isTRUE(all.equal(loss(w), - sum(reliaR::dgumbel(y, X %*% tail(w, -1), exp(w[1]), log = TRUE)))))
stopifnot(isTRUE(all.equal(gradient(w), numDeriv::grad(loss, w))))
