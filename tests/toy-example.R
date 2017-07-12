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
