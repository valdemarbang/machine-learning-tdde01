# produce the training data in dat
x <- runif(500, -4, 4)
y <- sin(x)
dat <- cbind(x, y)
plot(dat)
gamma <- 0.01
h <- function(z) {
  # activation function (sigmoid)
  return(1 / (1 + exp(-z)))
}
hprime <- function(z) {
  # derivative of the activation function (sigmoid)
  return(h(z) * (1 - h(z)))
}
yhat <- function(x) {
  # prediction for point x
}
MSE <- function() {
  # mean squared error
}
# initialize parameters
w1 <- matrix(runif(2,-.1,.1), nrow = 2, ncol = 1)
b1 <- matrix(runif(2,-.1,.1), nrow = 2, ncol = 1)
w2 <- matrix(runif(2,-.1,.1), nrow = 1, ncol = 2)
b2 <- matrix(runif(1,-.1,.1), nrow = 1, ncol = 1)
res <- NULL

for (i in 1:100000) {
  if (i %% 1000 == 0) {
    res <- c(res, MSE())
  }
  # forward propagation
  q0 <- dat[j, 1]
  z1 = w
  q1 <- h(i)
  
  j <- sample(1:nrow(dat), 1)
  # backward propagation
  
  # parameter updating
}
plot(res, type = "l")
plot(dat)
points(dat[, 1], lapply(dat[, 1], yhat))


# Read the exercise entirely before starting. You are asked to implement the backpropagation algorithm
# for training a neural network for regression as it appears in the course textbook and slides. You can find
# the pseudocode below. The neural network has one hidden layer with two units. W denotes weights, b
# denotes intercepts, z denotes activation units, q denotes hidden units (i.e. the result of applying the
# activation function h to z), J denotes the squared error, and gamma denotes the learning rate. The
# superscript indicates the layer (0=input layer, 1=hidden layer, 2=output layer). All products are matrix
# products (%*% in R), except the one indicated with that is element-wise product (* in R). Note the use
# of matrix transposition in some steps (t() in R). Comment your code.


