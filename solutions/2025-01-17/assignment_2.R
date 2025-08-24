#########################
# Kernel methods - TDDE01 - 6p
# This script:
# 1. Generates 1D synthetic data for two classes (mixture distributions)
# 2. Implements simple (teacher-style) kernel density estimates using first 800 samples (train)
# 3. Validates bandwidth h on next 100 samples (validation)
# 4. Retrains using 900 samples (train+validation) and estimates test accuracy on last 100
# 5. Performs a simple 2-fold NN cross-validation, then trains final NN
#########################

set.seed(123456789)

N_class1 <- 1000
N_class2 <- 1000

# Generate class 1:
# For each point: with prob 0.3 draw from N(15,3^2), else from N(4,2^2)
data_class1 <- NULL
for(i in 1:N_class1){
  a <- rbinom(n = 1, size = 1, prob = 0.3)               # mixture component indicator
  b <- rnorm(n = 1, mean = 15, sd = 3) * a +
       (1-a) * rnorm(n = 1, mean = 4, sd = 2)
  data_class1 <- c(data_class1,b)
}

# Generate class 2:
# With prob 0.4 draw from N(10,5^2), else from N(15,2^2)
data_class2 <- NULL
for(i in 1:N_class2){
  a <- rbinom(n = 1, size = 1, prob = 0.4)
  b <- rnorm(n = 1, mean = 10, sd = 5) * a +
       (1-a) * rnorm(n = 1, mean = 15, sd = 2)
  data_class2 <- c(data_class2,b)
}

# ----- Kernel density estimates (training on first 800 points) -----
# NOTE: Standard KDE formula is (1/(n*h)) * sum K((t - xi)/h).
# The teacher formula given in assignment omits the 1/h factor, so we follow that specification.

conditional_class1 <- function(t, h){
  d <- 0
  for(i in 1:800)
    d <- d + dnorm((t - data_class1[i]) / h)   # accumulate kernel contributions
  return (d / 800)                              # average (teacher version)
}

conditional_class2 <- function(t, h){
  d <- 0
  for(i in 1:800)
    d <- d + dnorm((t - data_class2[i]) / h)
  return (d / 800)
}

# ----- Posterior P(Class=1 | t) with equal implicit priors (800/1600) -----
prob_class1 <- function(t, h){
  prob_class1 <- conditional_class1(t, h) * 800 / 1600
  prob_class2 <- conditional_class2(t, h) * 800 / 1600
  return (prob_class1 / (prob_class1 + prob_class2))
}

# ----- Bandwidth selection via validation (points 801:900 in each class) -----
foo <- NULL
for(h in seq(0.1, 5, 0.1)){
  # Classification rule: predict class 1 if posterior > 0.5
  acc <- ( sum(prob_class1(data_class1[801:900], h) > 0.5) +
           sum(prob_class1(data_class2[801:900], h) < 0.5) ) / 200
  foo <- c(foo, acc)
}
plot(seq(0.1, 5, 0.1), foo)                     # validation accuracy vs h

max(foo)                                        # best validation accuracy
which(foo == max(foo)) * 0.1                    # corresponding h

# ----- Retrain densities using 900 (train+validation) points, test on last 100 -----

conditional_class1 <- function(t, h){
  d <- 0
  for(i in 1:900)
    d <- d + dnorm((t - data_class1[i]) / h)
  return (d / 900)
}

conditional_class2 <- function(t, h){
  d <- 0
  for(i in 1:900)
    d <- d + dnorm((t - data_class2[i]) / h)
  return (d / 900)
}

prob_class1 <- function(t, h){
  # Priors now 900/1800 each (still equal)
  prob_class1 <- conditional_class1(t, h) * 900 / 1800
  prob_class2 <- conditional_class2(t, h) * 900 / 1800
  return (prob_class1 / (prob_class1 + prob_class2))
}

h <- which(foo == max(foo)) * 0.1               # selected bandwidth
# Test accuracy on hold-out samples 901:1000 (100 per class)
(sum(prob_class1(data_class1[901:1000], h) > 0.5) +
 sum(prob_class1(data_class2[901:1000], h) < 0.5)) / 200

##########################
# Neural networks - TDDE01 - 4p
##########################

library(neuralnet)
set.seed(1234567890)

# Create regression data: y = sin(x) sampled at 50 random x values
Var <- runif(50, 0, 10)
trva <- data.frame(Var, Sin = sin(Var))
tr1 <- trva[1:25,]   # fold 1
tr2 <- trva[26:50,]  # fold 2

# ----- 2-fold cross-validation for NN generalization error -----
# Each network: 1 input -> 10 hidden neurons -> 1 output
# Weight init length 31 = (1+1)*10 (input+ bias to hidden) + (10+1)*1 (hidden + bias to output)

winit <- runif(31, -1, 1)
nn1 <- neuralnet(formula = Sin ~ Var, data = tr1, hidden = 10,
                 startweights = winit, threshold = 0.001, lifesign = "full")

winit <- runif(31, -1, 1)
nn2 <- neuralnet(formula = Sin ~ Var, data = tr2, hidden = 10,
                 startweights = winit, threshold = 0.001, lifesign = "full")

# Cross-predictions (each net evaluated on the other fold)
aux1 <- predict(nn1, tr2)
aux2 <- predict(nn2, tr1)

# Symmetric CV error (sum of MSE * 1/2 per fold)
sum((tr2[,2] - aux1)^2) / 2 + sum((tr1[,2] - aux2)^2) / 2

# ----- Final network trained on all data for deployment -----
winit <- runif(31, -1, 1)
nn1 <- neuralnet(formula = Sin ~ Var, data = rbind(tr1, tr2), hidden = 10,
                 startweights = winit, threshold = 0.001, lifesign)