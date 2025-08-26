set.seed(1234567890)
spam <- read.csv2("spambase.csv")
ind <- sample(1:nrow(spam))
spam <- spam[ind, c(1:48, 58)]
h <- 1
beta <- 0
M <- 500
N <- 500 # number of training points

gaussian_k <- function(x, y, h) {
  # Gaussian kernel
  d <- dist(rbind(x, y)) # Euclidean distance
  exp(-(d^2) / (2 * h^2))
}
SVM <- function(sv, i, alpha, b) {
  # SVM on point i with support vectors sv
  
  # Note that the labels in spambase.csv are 0/1 and SVMs need -1/+1. Then, use 2*label-1
  # to convert from 0/1 to -1/+1
  # Do not include the labels when computing the Euclidean distance between the point i
  # and each of the support vectors. This is the distance to use in the kernel function
  # You can use dist() to compute the Euclidean distance
  
  xi <- as.numeric(spam[i, -ncol(spam)])
  yi <- 0
  for (m in seq_along(sv)) {
    xm <- as.numeric(spam[sv[m], -ncol(spam)])
    tm <- spam[sv[m], ncol(spam)]
    yi <- yi + alpha[m] * tm * gaussian_k(xi, xm, h)
  }
  return(yi + b)
}
errors <- 1
errorrate <- vector(length = N)
errorrate[1] <- 1
for (i in 2:N) {
  xi <- as.numeric(spam[i, -ncol(spam)])
  ti <- spam[i, ncol(spam)]
  
  yi <- SVM(sv, i, alpha, b)
  pred <- sign(yi)
  
  # Update if margin violated
  if (ti * yi <= beta) {
    sv <- c(sv, i)
    alpha <- c(alpha, 1)
  }
  
  # Budget control
  if (length(sv) > M) {
    scores <- numeric(length(sv))
    for (m in seq_along(sv)) {
      xm <- as.numeric(spam[sv[m], -ncol(spam)])
      tm <- spam[sv[m], ncol(spam)]
      ym <- SVM(sv, sv[m], alpha, b)
      scores[m] <- tm * (ym - alpha[m] * tm * gaussian_k(xm, xm, h))
    }
    drop_idx <- which.max(scores)
    sv <- sv[-drop_idx]
    alpha <- alpha[-drop_idx]
  }
  
  # Track errors
  if (pred != ti)
    errors <- errors + 1
  errorrate[i] <- errors / i
}
plot(errorrate[seq(from = 1, to = N, by = 10)], ylim = c(0.2, 0.4), type = "o")
length(sv)
errorrate[N]
