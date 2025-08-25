# Scale all variables except of Death and then use these scaled features to perform PCA. Report
# how much of the total variation is captured by the first two principal components. Report an
# equation of the first principal component in terms of the scaled original variables. Finally, report
# a scatter plot of the data in the coordinate system (PC1, PC2) where observations are colored by
# Death and comment which of these PCs is best in discriminating between the survival groups.

### Library ###
library(dplyr)
library(caret)
library(ggplot2)

### Task 1 ###
data_frame = read.csv("women.csv")
df_no_death = select(data_frame, !Death)
scaler = preProcess(df_no_death)
data_scaled = predict(scaler, df_no_death)
pca_res = princomp(data_scaled)

data_scaled$Death = data_frame$Death
pca_res
# 0.1350596 (13.5%) comp1 and 0.1291777 (12.9%) comp2
# 0.2642373 (26.4%) for comp1 and comp2 total.

pca_res$loadings[, 1]
# comp1 = 0.37697839*Age - BMI0.552 * BloodPressure - 0.343 * Glucose - 0.472 * WhiteBloodCell + 0.484 * RedBloodCell

PCA1 <- (pca_res$scores)[, 1]
PCA2 <- (pca_res$scores)[, 2]

df_plot <- data.frame(PC1 = PCA1,
                      PC2 = PCA2,
                      Death = factor(data_scaled$Death))
ggplot(df_plot, aes(x = PC1, y = PC2, color = Death)) +
  geom_point() +
  labs(title = "PC1 vs PC2 scores", x = "PC1", y = "PC2") +
  theme_minimal()

# The first two PCs together capture around 26% of the variance. From the scatterplot there is some seperation
# for high PC1 or high PC2 which results in more deaths.

### Task 2 ###
# Split the original dataset without Death variable into training and test sets (50/50) and then
# consider the training sets that have first 100 observations, first 200 observations, …, first 2400
# observations, 2500 observations. For each of these data sets, estimate a) a decision tree with
# 10 leaves and b) K-nearest neighbor with K=10 where Cholestrol is used as target and all
# remaining variables as features. Compute MSE for each of the training and test sets and each of
# the two model types and then plot a dependence of training and test errors on the size of the
# training set a) for decision tree models b) for K-nearest neighbor models. Comment on the
# trends you observe in the decision tree plot and the theoretical reasons behind these trends. By
# comparing the two plots, report which of the two models can be considered as more complex
# one for these data and why.

# Split the dataset into training (50%), and test (50%).
n = dim(df_no_death)[1] # Total nr of rows in dataset
id = sample(1:n, floor(n * 0.5)) # Randomly selects 50 % rows for training
id1 = setdiff(1:n, id) # Remaining rows
data_train = df_no_death[id, ] # Assign rows to training set
data_test = df_no_death[id1, ] # Set remaining rows as test data

library(tree)
library(kknn)

obs_seq = seq(from = 100, to = 2500, by = 100)
tree_train_mse <- c()
tree_test_mse <- c()
kknn_train_mse <- c()
kknn_test_mse <- c()

for (i in obs_seq) {
  obs_data_train = data_train[1:i, ]
  
  # --- Decision Tree ---
  tree_model = tree(Cholestrol ~ ., data = data_train[1:i, ])
  
  leaves = sum(tree_model$frame$var == "<leaf>")
  
  if (leaves > 10) {
    pruned_tree = prune.tree(tree_model, best = 10)
  } else {
    pruned_tree = tree_model  # använd originalträdet
  }
  
  # Training error
  pred_train = predict(pruned_tree, newdata = obs_data_train)
  tree_train_mse = c(tree_train_mse, mean((obs_data_train$Cholestrol - pred_train)^2))
  
  # Test error
  pred_test = predict(pruned_tree, newdata = data_test)
  tree_test_mse = c(tree_test_mse, mean((data_test$Cholestrol - pred_test)^2))
  
  # --- KNN ---
  kknn_model = kknn(Cholestrol ~ .,
                    train = obs_data_train,
                    test = obs_data_train,
                    k = 10)
  kknn_train_pred = predict(kknn_model)
  kknn_train_mse = c(kknn_train_mse, mean((
    obs_data_train$Cholestrol - kknn_train_pred
  )^2))
  
  kknn_model_test = kknn(Cholestrol ~ .,
                         train = obs_data_train,
                         test = data_test,
                         k = 10)
  kknn_test_pred = predict(kknn_model_test)
  kknn_test_mse = c(kknn_test_mse, mean((data_test$Cholestrol - kknn_test_pred)^2))
}

# --- Plots ---
plot(
  obs_seq,
  tree_train_mse,
  type = "l",
  col = "blue",
  xlab = "Training set size",
  ylab = "MSE",
  ylim = c(200, 2000),
  main = "Decision Tree"
)
lines(obs_seq, tree_test_mse, col = "red")
legend(
  "topright",
  legend = c("Train", "Test"),
  col = c("blue", "red"),
  lty = 1
)

plot(
  obs_seq,
  kknn_train_mse,
  type = "l",
  col = "blue",
  xlab = "Training set size",
  ylab = "MSE",
  ylim = c(500, 2000),
  main = "KNN"
)
lines(obs_seq, kknn_test_mse, col = "red")
legend(
  "topright",
  legend = c("Train", "Test"),
  col = c("blue", "red"),
  lty = 1
)
# Decision Tree:
# Train starts with a lower mse and get a higher one the bigger the dataset
# Which means small dataset, tree can fit them quite well => low training error
# Test starts with a high mse and gets a lower one the bigger the dataset
# Small datasets, tree overfits and give high test error
# But with larger dataset for training the test error goes down.

# KKNN:
# The test stays solid with a high MSE
# The train also stays pretty solid with a lower MSE
# In total showing higher bias and lower variance.

# The decision tree is more complex here because its error curve clearly depends on training size.
# Showing more variance => which means its more complex in the bias/variance ratio.


### Task 3 ###
# Split original data into training and test sets (50/50) and estimate a Ridge classification model
# where Death is target and all other variables are features so that optimal penalty parameter is
# selected by the cross-validation from the training data. Report the optimal penalty factor, and
# confusion matrix for the test dataset. Use also the following cost matrix to update the
# confusion matrix for the test data:
# and comment on what kind of changes in the confusion matrix you observed and why they
# happened.
library(glmnet)

n = dim(data_frame)[1] # Total nr of rows in dataset
id = sample(1:n, floor(n * 0.5)) # Randomly selects 50 % rows for training
id1 = setdiff(1:n, id) # Remaining rows
data_train = data_frame[id, ] # Assign rows to training set
data_test = data_frame[id1, ] # Set remaining rows as test data

# Create the ridge model by setting alpha = 0.
cv_ridge <- cv.glmnet(as.matrix(select(data_train, !Death)),
                      data_train$Death,
                      alpha = 0,
                      family = "binomial")
opt_penalty_param = cv_ridge$lambda.min
opt_penalty_param # 0.04998361

predictions_test = predict(cv_ridge,
                      s = opt_penalty_param,
                      newx = as.matrix(select(data_test, !Death)),
                      type = "class")

conf_matrix = table(True = data_test$Death, Preds = predictions_test)
# > conf_matrix
# predictions_test
# True   0   1
# 0 262 369
# 1 167 452


cost_matrix = matrix(
  c(0, 1, 10, 0),
  byrow = TRUE,
  nrow = 2,
  dimnames = list(c("True 0", "True 1"), c("Pred 0", "Pred 1"))
)

adjusted_matrix <- conf_matrix * cost_matrix # Matrices are already aligned 2x2 * 2x2, if it 
                                              # was across a class of probabilities then we would use %*%
adjusted_matrix
# > adjusted_matrix
# predictions_test
# True    0    1
# 0    0  369
# 1 1670    0
# It punishes heavily when the Death is True label but the predictions is 0 with a (10x)
# When the Death is false (0) and the predictions is true is gives a smaller penalty of (1x)
# It ignores the cost of the other ones multipled by 0