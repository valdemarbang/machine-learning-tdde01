########################## Packages ##########################
library(tree)
library(rpart)

########################## PART 1 ##########################
# Assume that Sex is the target variable in the modeling and all crab measurements are the
# features, and perform model selection by using decision trees and the hold-out method (60%
# train/ 40% test). Present a plot showing dependence of the training and test-cross entropies on
# the amount of leaves in the tree and report how many leaves the optimal tree has. Motivate
# your model choice. Why is cross-entropy a reasonable metric here? How many features are
# selected by the optimal tree? Why do tree branches in the tree with 7 leaves from some parent
# node lead to the leaves with exactly same labels? (3p)
data_frame = read.csv("australian-crabs.csv", stringsAsFactors = TRUE)

# Split the dataset into training (60%), (40%) train sets
n = dim(data_frame)[1] # Total nr of rows in dataset
set.seed(12345)
id = sample(1:n, floor(n * 0.6))
data_train = data_frame[id, ]
data_test  = data_frame[-id, ]

tree = tree(formula = sex ~ ., data = data_train)

# Init vector with length 50 to store score
train_score = rep(0, 50)
test_score = rep(0, 50)

# Calculate the deviance on both datasets and save to the score vector.
for (leaves in 2:50) {
  pruned_tree = prune.tree(tree, best = leaves)
  
  prediction = predict(pruned_tree, newdata = data_test, type = "tree")
  
  train_score[leaves] = deviance(pruned_tree)
  test_score[leaves] = deviance(prediction)
}

# Graph of the dependence of deviance for the datasets on the number of leaves
plot(
  2:50,
  train_score[2:50],
  type = "b",
  col = "red",
  ylim = c(min(train_score[2:50]), max(test_score[2:50])),
  xlim = c(2, 50),
  main = "Optimal tree depth",
  ylab = "Deviance",
  xlab = "Number of leaves"
)
points(2:50, test_score[2:50], type = "b", col = "blue")
legend("topright", c("train data", "test data"), fill = c("red", "blue"))
# Optimal number of leaves that minimize training deviance.
optimal_test = which.min(test_score[2:50]) + 1
optimal_test
optimal_tree = prune.tree(tree, best = optimal_test)
# Display optimal tree
plot(optimal_tree)
text(optimal_tree, pretty = 0)
optimal_tree
summary(optimal_tree)
# Motivate your model choice. Why is cross-entropy a reasonable metric here?
# Its a proper scoring metric, it rewards well-calibrated probabilities, not just correct labels.
# Overconfident wrong predictions are penalized heavily. It works well with class imbalance and
# compares models without picking a threshold.

# How many features are selected by the optimal tree?
# 2 Variables actually used in tree construction: [1] "RW" "CW"

# Why do tree branches in the tree with 7 leaves from some parent node lead to the leaves with exactly same labels?
# A split can lower the cross-entropy even if both children predict the same majority class.

########################## PART 2 ##########################

# Assume that Sex is the target variable in the modeling and all measurements are the features,
# and compute a logistic regression model by using the entire dataset.
data_frame = read.csv("australian-crabs.csv", stringsAsFactors = TRUE)
lrm = glm(formula = sex ~ .,
          data = data_frame,
          family = "binomial")

# Compute the predicted probabilities for the first observation in the dataset.
# Use output from this model to compute how much these probabilities change if we set
# the parameters corresponding to CW and BD to zero.
predictions_lrm = predict(lrm, newdata = data_frame[1, ], type = "response")
predictions_lrm
# 0.9999879

single_df = data_frame[1, ]
single_df$CW = 0
single_df$BD = 0
predictions_lrm = predict(lrm, newdata = single_df, type = "response")
predictions_lrm
# 2.220446e-16

# Report mathematical calculations that were used for computing the updated probabilities.
single_df
lrm$coefficients
# The data from single dataframe multiplied with the coefficients from the logistic regression model
# gives the prediction probability.
# The probabilistic model is P(Class = Sex) = 1/ (1 + exp(z))
# Where z = 25.3251579 + 2.2168800 * speciesOrange - 0.7203229 * index - 34.1936982 * FL - 34.1936982 * RW + 4.4249286 * CW + 5.4552912 * BD

# Compute F1-score (positive class=Male) for the entire dataset by using the following loss matrix:

loss_matrix = matrix(
  c(0, 1, 10, 0),
  byrow = TRUE,
  nrow = 2,
  dimnames = list(c("True Male", "True Female"), c("Pred Male", "Pred Female"))
)

predictions_lrm = predict(lrm, type = "response")

# Probabilities
probs = cbind(
  "Pred Male" = predictions_lrm,
  # Positive class male
  "Pred Female" = 1 - predictions_lrm
) # Else Female

# Expected losses per sample, transpose to get same row and column types.
losses = probs %*% t(loss_matrix)

# Pick class with smallest expected loss
predicted_classes = apply(losses, 1, which.min)
predicted_classes = ifelse(predicted_classes == 1, "Male", "Female")

# Confusion matrix
conf_matrix = table(True = data_frame$sex, Predicted = predicted_classes)

accuracy_and_f1 = function(conf_matrix) {
  TP = conf_matrix["Male", "Male"] # True positives
  TN = conf_matrix["Female", "Female"] # True negatives
  FP = conf_matrix["Female", "Male"] # False positives
  FN = conf_matrix["Male", "Female"] # False negatives
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  accuracy = (TN + TP) / (TN + FP + TP + FN)
  f1_score = 2 * (precision * recall) / (precision + recall)
  return(c(accuracy = accuracy, f1_score = f1_score))
}

result = accuracy_and_f1(conf_matrix)
print(result["accuracy"])
print(result["f1_score"])

# Is F1 score or the accuracy a more relevant metric for this dataset? Motivate your answer.
# Both have similar scoring for both metrics, so I would say the dataset is pretty balanced. So the accuracy is more relevant when the
# dataset is balanced and F1-Score is more relevant when the dataset is unbalanced. So in this case the Accuracy is more relevant.


# Implement a minus log-likelihood function in R that describes this model as a function of the parameters.
# Use the BFGS optimization method with starting point (0,0) to compute and report the optimal parameters,
# using the entire dataset. Finally, compute the prediction interval for the first observation in the dataset.
# Model: RW | FL ~ N(mu = w0 + w1*FL, sigma^2 = 0.1*FL - 0.5)
y <- data_frame$RW
x <- data_frame$FL
X <- cbind(1, x)            # like trainS_without_y_matrix in your help paper
s2 <- 0.1 * x - 0.5         # per-sample variance

# Negative log-likelihood (heteroskedastic Gaussian) in "help paper" style
neg_logLik_homo <- function(theta, s2, X, y) {
  n <- nrow(X)
  r <- y - as.vector(X %*% theta)
  #-0.5 * n * log(2*pi) - 0.5 * n * log(s2) - 0.5 * sum(r^2) / s2
  0.5 * nrow(X) * log(2 * pi) + 0.5 * sum(log(s2)) + 0.5 * sum((r^2) / s2)
  # equivalently: 0.5 * sum(log(2 * pi * s2) + (r^2) / s2)
}

# Optimize with BFGS starting at (0, 0)
opt <- optim(
  par = c(0, 0),
  fn = neg_logLik_homo,
  method = "BFGS",
  X = X, y = y, s2 = s2
)

# Optimal parameters
w0_hat <- opt$par[1]
w1_hat <- opt$par[2]
opt$par
cat("w0_hat =", w0_hat, " w1_hat =", w1_hat, "\n")

# 95% prediction interval for the first observation (given FL[1])
x1 <- x[1]
mu1 <- w0_hat + w1_hat * x1
s2_1 <- 0.1 * x1 - 0.5

pi_1 <- mu1 - 1.95 * sqrt(s2_1)
pi_2 <- mu1 + 1.95 * sqrt(s2_1)
cat("Pred. mean =", mu1, " 95% PI = [", pi_1, ", ", pi_2, "]\n")
