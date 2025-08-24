# Scale the data and apply cross-validation compute LASSO regression models which predict Area
# from the other numerical variables (i.e exclude Diagnosis). 

library(glmnet)
library(dplyr)
library(caret)

set.seed(12345)

data <- read.csv("wdbc.csv")

predictors_raw <- select(data, -ID, -Diagnosis, -AreaMean)
y <- data$AreaMean

# Scale ONLY predictors (not y)
scaler <- preProcess(predictors_raw)
X_scaled <- predict(scaler, predictors_raw)

model_lasso <- cv.glmnet(
  x = as.matrix(X_scaled),
  y = y,
  alpha = 1,
  family = "gaussian"
)

# Provide a plot showing dependence of the cross-validation error and its uncertainty on the value of the penalty parameter.
plot(model_lasso, xvar = "lambda")

# How many features are selected by the optimal model? 
coef_opt <- coef(model_lasso, s = "lambda.min")
n_features_opt <- sum(coef_opt[-1] != 0)  # exclude intercept
cat("Number of non-zero predictors at lambda.min:", n_features_opt, "\n")

# Report an equation that shows how the target variable can be predicted from the features when log(λ) = -2.
lambda_target <- exp(-2)
# Get coefficients at chosen lambda
coef_target <- coef(model_lasso, s = lambda_target)
# Keep only non-zero coefficients
non_zero <- which(coef_target != 0)
names_keep <- rownames(coef_target)[non_zero]
vals_keep <- as.numeric(coef_target[non_zero])
# Build equation
intercept <- round(vals_keep[1], 4)
terms <- paste(round(vals_keep[-1], 4), "*", names_keep[-1])
equation <- paste("AreaMean_hat =", intercept, "+", paste(terms, collapse = " + "))
cat("Equation at log(lambda) = -2:\n", equation, "\n")

# Is the model for log(𝜆) = −2 statistically significantly different from the optimal model? (3p)
cvm     <- model_lasso$cvm      # mean cross-validation error for each lambda
cvsd    <- model_lasso$cvsd     # standard error of CV error for each lambda
lambdas <- model_lasso$lambda   # lambda values tested

# Find the index of the optimal lambda
idx_opt <- which(model_lasso$lambda == model_lasso$lambda.min)

# Find the index of the lambda we are interested in (log(lambda) = -2)
idx_target <- which.min(abs(lambdas - exp(-2)))

# Print the CV errors
cat("Optimal lambda:\n")
cat("  lambda =", lambdas[idx_opt],
    "\n  CV error =", cvm[idx_opt], "±", cvsd[idx_opt], "\n\n")

cat("Target lambda (log(lambda) = -2):\n")
cat("  lambda =", lambdas[idx_target],
    "\n  CV error =", cvm[idx_target], "±", cvsd[idx_target], "\n")

# Divide the original (unscaled) data into training and validation (50/50) and compute a logistic
# regression model in which Diagnosis is predicted from the remaining variables. Compute the
# training and test misclassification errors and comment if the model seems to be overfitted.
# Assuming “M” to be the positive class, compute precision values for the test data with the
# following loss matrices and comment why precision changed in that direction with L1 and L2 (3p):

set.seed(12345)
data <- read.csv("wdbc.csv")
data$Diagnosis <- ifelse(data$Diagnosis == "M", 1, 0)

n = dim(data)[1]

id = sample(1:n, floor(n * 0.5))
id1 = setdiff (1:n, id) # Remaining rows
data_train = data[id,] # Assign rows to training set
data_val = data[id1,] # Set remaining rows as test data

lrm <- glm(formula = Diagnosis ~ ., data = data_train , family = "binomial")

misclass=function(X,X1){
  n=length(X)
  return(1-sum(diag(table(X,X1)))/n)
}
misclass(data_train$Diagnosis, predict(lrm, newdata = data_train, type = "response")>0.5)
misclass(data_val$Diagnosis, predict(lrm, newdata = data_val, type = "response")>0.5)

# > misclass(predictions_train, data_train$Diagnosis)
# [1] 0
# > misclass(predictions_val, data_val$Diagnosis)
# [1] 0.05964912
# Model doesnt seem to be overfitted because it scores well on unseen data such as the validation dataset, with very low misclass rate.

probs_val <- predict(lrm, newdata = data_val, type = "response")

predict_with_loss <- function(probs, loss_matrix) {
  # probs = P(Y=1|X), so prob for class B = 1 - probs
  losses_B <- (1 - probs) * loss_matrix["True B", "Predict B"] +
    probs * loss_matrix["True M", "Predict B"]
  losses_M <- (1 - probs) * loss_matrix["True B", "Predict M"] +
    probs * loss_matrix["True M", "Predict M"]
  
  ifelse(losses_M < losses_B, 1, 0)  # Predict M if loss smaller
}

# Prediction under L1
pred_val_L1 <- predict_with_loss(probs_val, L1)
conf_matrix_L1 <- table(data_val$Diagnosis, pred_val_L1)

# Prediction under L2
pred_val_L2 <- predict_with_loss(probs_val, L2)
conf_matrix_L2 <- table(data_val$Diagnosis, pred_val_L2)

# Precision function
precision <- function(conf_matrix) {
  TP = conf_matrix[2, 2]
  FP = conf_matrix[1, 2]
  TP / (TP + FP)
}

res_1 = precision(conf_matrix_L1)
res_2 = precision(conf_matrix_L2)

res_1
res_2

# Assume now the following classification model: 𝑦𝑦�(𝑥𝑥) = 𝑠𝑠𝑃𝑃𝑠𝑠𝑠𝑠�𝑓𝑓(𝑥𝑥)� where 𝑓𝑓(𝑥𝑥) = 𝑤𝑤𝑇𝑇𝑥𝑥, where
# 𝑦𝑦� is the predicted diagnosis  (“-1”  corresponds to “B” and “1” corresponds to “M”), 𝑥𝑥 are all
# other variables in the data and 𝑤𝑤 is the set of parameters. Implement a cost function in R
# depending on argument 𝑤𝑤 that uses this model, training data from step 2 and the Hinge loss
# function. Optimize the cost function from the starting point (0,..., 0) by the BFGS method and
# estimate training and test misclassification errors for the optimal model. Report the estimated
# predictive equation. Compare the quality of this model with the quality of the logistic model
# from step 2. (4p)

# Hinge loss function
hinge_loss <- function(w, X, y) {
  f_x <- X %*% w              # f(x) för alla observationer
  margins <- y * f_x          # y * f(x)
  loss <- ifelse(margins <= 1, 1 - margins, 0)
  return(sum(loss))           # summera över alla
}

# Prediction function
predict_hinge <- function(w, X) {
  preds <- sign(X %*% w)
  return(preds)
}

# Example usage
# X_train: matrix of predictors (rows = samples, cols = features)
# y_train: vector of labels in {-1, 1}

X_train <- as.matrix(select(data_train, -Diagnosis))
y_train <- ifelse(data_train$Diagnosis == "M", 1, -1)

X_test <- as.matrix(select(data_val, -Diagnosis))
y_test <- ifelse(data_val$Diagnosis == "M", 1, -1)

# korrekt startvektor
w0 <- rep(0, ncol(X_train))

opt <- optim(w0, hinge_loss, X = X_train, y = y_train, method = "BFGS")
opt
# Optimal weights
w_opt <- opt$par

# Training error
train_preds <- predict_hinge(w_opt, X_train)
train_error <- mean(train_preds != y_train)

# Test error
test_preds <- predict_hinge(w_opt, X_test)
test_error <- mean(test_preds != y_test)

cat("Train error:", train_error, "\n")
cat("Test error:", test_error, "\n")
cat("Optimal weights:", w_opt, "\n")