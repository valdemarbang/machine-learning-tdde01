set.seed(12345)

### library ###
library(dplyr)
library(caret)
library(kknn)

### Task 1 ###
data_frame = read.csv("glass.csv")
df_no_class = select(data_frame, !Class)

scaler = preProcess(df_no_class)
df_no_class_scaled = predict(scaler, df_no_class)
pca_res = princomp(df_no_class_scaled)

summary(pca_res)
# How much variation is explained by the first two components?
# Comp1 0.1623562, Comp2 0.1390911, total= 0.3014473, ~30%

# If we shall treat this operation as a data compression
# task where we keep only first two principal components but might want to restore the data in
# the original feature scale, compare how many numbers need to be stored in the original data,
# and how many numbers we need to keep in the compressed data.
features_pca12 = pca_res$loadings[, 1:2]
features_pca12
# In the original data we got n = 120 samples, p = 9 chemical components
# Original: 120*9 = 1080 numbers
# Compressed: (120 * 2) + (9 * 2) = 240 + 18 = 258 numbers


### Task 2 ###
pca_raw <- princomp(df_no_class)
loadings(pca_raw)
#   Comp.1 Comp.2 Comp.3 Comp.4 Comp.5 Comp.6 Comp.7 Comp.8 Comp.9
# RI                                                          1.000
# Na         0.824  0.470  0.309
# Mg         0.462 -0.264 -0.840
# Al                             -0.993
# Si  0.998
# K                                     -0.992
# Ca         0.318 -0.839  0.434
# Ba                                            0.996
# Fe                                                   0.998
# Report equations of the first two principal components in terms of centered original features.
# Comp1 = 0.998 * Si
# Comp2 = 0.824 * Na + 0.462 * Mg + 0.318 * Ca

# Report also equations of original features in terms of the first two principal components.
# Si = Comp1 / 0.998

# Finally, comment on which variables contribute mostly to each of the two principal components, respectively.
# Si towards Comp1 and Na towards Comp2


### Task 3 ###
n = dim(data_frame)[1]
set.seed(12345)
id = sample(1:n, floor(n * 0.5))
data_train = data_frame[id, ]
data_test = data_frame[-id, ]

scaler = preProcess(select(data_train, !Class))

data_train_s = predict(scaler, select(data_train, !Class))
# data_train_s$Class = data_train$Class

data_test_s = predict(scaler, select(data_test, !Class))
# data_test_s$Class = data_test$Class

cost_function <- function(w, x, y) {
  fx <- as.vector(x %*% w)
  loss <- mean(exp(-y * fx))
  
  return(loss)
}

w0 <- rep(0, ncol(data_train_s))

# Optimize
res_train <- optim(
  w0,
  fn = cost_function,
  x = as.matrix(data_train_s),
  y = data_train$Class,
  method = "BFGS",
  control = list(maxit = 1000, trace = 1, REPORT = 1)
)

res_test <- optim(
  w0,
  fn = cost_function,
  x = as.matrix(data_train_s),
  y = data_train$Class,
  method = "BFGS",
  control = list(maxit = 1000, trace = 1, REPORT = 1)
)

w_opt <- res_train$par
fx_test <- as.vector(as.matrix(data_test_s) %*% w_opt)
y_pred <- sign(fx_test)

conf_matrix = table(data_test$Class, y_pred)
conf_matrix
