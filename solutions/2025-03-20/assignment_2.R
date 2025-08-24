set.seed(12345)
N_class1 <- 1000
N_class2 <- 1000
# Generate class 1:
# For each point: with prob 0.3 draw from N(15,3^2), else from N(4,2^2)
data_class1 <- NULL
for (i in 1:N_class1) {
  a <- rbinom(n = 1, size = 1, prob = 0.3) # mixture component indicator
  b <- rnorm(n = 1, mean = 15, sd = 3) * a +
    (1 - a) * rnorm(n = 1, mean = 4, sd = 2)
  data_class1 <- c(data_class1, b)
}

# Generate class 2:
# With prob 0.4 draw from N(10,5^2), else from N(15,2^2)
data_class2 <- NULL
for (i in 1:N_class2) {
  a <- rbinom(n = 1, size = 1, prob = 0.4)
  b <- rnorm(n = 1, mean = 10, sd = 5) * a +
    (1 - a) * rnorm(n = 1, mean = 15, sd = 2)
  data_class2 <- c(data_class2, b)
}

# Split sizes
n_train <- 700
n_val   <- 300
train_ix <- 1:n_train
val_ix   <- (n_train + 1):(n_train + n_val)

# ----- KDE on training portion (teacher’s formula; 1/h omitted but cancels) -----
conditional_class1 <- function(t, h) {
  d <- 0
  for (i in train_ix)
    d <- d + dnorm((t - data_class1[i]) / h)
  d / n_train
}
conditional_class2 <- function(t, h) {
  d <- 0
  for (i in train_ix)
    d <- d + dnorm((t - data_class2[i]) / h)
  d / n_train
}

# ----- Posterior with priors from training counts -----
prob_class1 <- function(t, h) {
  p1 <- conditional_class1(t, h)
  p2 <- conditional_class2(t, h)
  prior1 <- n_train / (n_train + n_train)     # equals 0.5 here
  prior2 <- 1 - prior1
  (p1 * prior1) / ((p1 * prior1) + (p2 * prior2) + .Machine$double.eps)
}

# ----- Validation accuracy vs h (300 per class) -----
hs <- seq(0.1, 5, 0.1)
val_acc  <- numeric(length(hs))
train_acc <- numeric(length(hs))

for (k in seq_along(hs)) {
  h <- hs[k]
  # validation accuracy (300 per class)
  val_acc[k] <- (sum(prob_class1(data_class1[val_ix], h) > 0.5) +
                   sum(prob_class1(data_class2[val_ix], h) < 0.5)) / (2 * n_val)
  # training accuracy (700 per class)
  train_acc[k] <- (sum(prob_class1(data_class1[train_ix], h) > 0.5) +
                     sum(prob_class1(data_class2[train_ix], h) < 0.5)) / (2 * n_train)
}

plot(hs, val_acc, type = "l", xlab = "h", ylab = "Accuracy",
     main = "KDE classifier: effect of bandwidth h", col = "red", lwd = 2)
lines(hs, train_acc, col = "blue", lwd = 2)
legend("bottomright", c("Validation", "Training"), col = c("red","blue"), lwd = 2, bty = "n")


### Neural Networks (2 points) ###

library(neuralnet)
set.seed(1234567890)

x1 <- runif(1000, -1, 1)
x2 <- runif(1000, -1, 1)
tr <- data.frame(x1,x2, y=x1 - x2)

winit <- runif(9, -1, 1) # weight init
nn<-neuralnet(formula = y ~ x1 + x2, data = tr, hidden = c(1), act.fct = "tanh")
plot(nn)

# So we got the w0 0.13555 for x1 and w1 -0.13553 for x2. Then we got the bias weight (blue line) -0.00293
# But the bias weight is so small so it doesnt really make a difference.
# Lets say x1=0.7 and x2=-0.2 then y=0.9 . We then get 0.7*0.13555+(-0.13553*-0.2)+small_bias_1=0.121985
# For the output node y_hat we then get 0.121985*7.4506+small_bias_2=Which gives us around 0.9
# The thing that makes sense is having x2 input multiplied with an negative weight and x1 multiplied with an positive weight
# This creates the y output which would be y= x1 - x2!
