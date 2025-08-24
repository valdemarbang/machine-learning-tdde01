setwd("~/Repos/machine-learning-tdde01/solutions/2025-01-17")

### library ###
library(dplyr)
library(caret)
library(kknn)

### Task 1 ###
data_frame = read.csv("lakesurvey.csv") #column , seperators
df_no_ph = select(data_frame, !pH & !LakeID)
scaler = preProcess(df_no_ph)
df_no_ph_s = predict(scaler, df_no_ph)

pca_res = princomp(na.omit(df_no_ph_s)) # na.omit(data) remove NA values
summary(pca_res)

# How much variation is explained by the first two principal components?
# Variance for component 1 and 2 is
# Proportion of Variance comp 1: 0.4306978 comp2: 0.1978562

loadings(pca_res) # Inspect loadings

features_pca_1 = pca_res$loadings[, 1]
abs(features_pca_1)

# Which features contribute to the first principal component mostly?
# The ones with absolute larger number contribute more to the first component.
# > pca_res$loadings[, 1]

# By assuming that we keep only first two principal
# components, report an equation showing how the unscaled Cond variable can be
# approximated from the first two principal components

df_no_ph_cc <- na.omit(df_no_ph) # 1) Remove rows with missing values
# Calculate the original mean for the 'Cond' variable before PCA
mean_cond_original <- mean(df_no_ph_cc$Cond)

pca_raw <- princomp(df_no_ph_cc, cor = FALSE)  # 2) Run PCA on raw data (no scaling, but centered by default)
# summary(pca_raw) # 3) Summary of variance explained
# loadings(pca_raw) # 4) Inspect loadings (in original units)
scores_raw <- pca_raw$scores[, 1:2]  # 5) Get PC scores for first two PCs (these are for centered data)

# 6) Get the loadings for the 'Cond' variable for the first two PCs
l_cond <- pca_raw$loadings["Cond", 1:2]

# 7) Approximate Cond from first two PCs, this result is for the *centered* Cond variable
Cond_hat_centered <- scores_raw %*% l_cond

# 8) Add back the original mean of 'Cond' to get the unscaled and uncentered approximation
Cond_hat_final <- Cond_hat_centered + mean_cond_original

# 9) View first 10 comparisons
head(cbind(Cond_true = df_no_ph_cc$Cond,
           Cond_hat_approx = Cond_hat_final), 10)

# The equation showing how the unscaled Cond variable can be approximated:
# Cond_approximated ≈ (lCond1 * PC1) + (lCond2 * PC2) + Mean(Cond_original)
# Where PC1 and PC2 are the scores (values from the principal co


### Task 2 ###

two_pca = pca_res$scores[, 1:2]
data_combined = cbind(two_pca, select(data_frame, pH)) # Dataset 3 variables

n = dim(data_frame)[1]
set.seed(12345)
id = sample(1:n, floor(n * 0.4))
data_train = data_frame[id, ]
combined_train = data_combined[id, ]

id1 = setdiff(1:n, id)
set.seed(12345)
id2 = sample(id1, floor(n * 0.3))
data_validation = data_frame[id2, ]
combined_validation = data_combined[id2, ]

id3 = setdiff(id1, id2)
data_test = data_frame[id3, ]
combined_test = data_combined[id3, ]

# Scale original dataset
scaler_original = preProcess(data_train)
data_train = predict(scaler_original, data_train)
data_validation = predict(scaler_original, data_validation)
data_test = predict(scaler_original, data_test)

# Scale the 3 variable combined dataset
scaler_combined = preProcess(combined_train)
combined_train = predict(scaler_combined, combined_train)
combined_validation = predict(scaler_combined, combined_validation)
combined_test = predict(scaler_combined, combined_test)

kknn_mse = function(train_data, test_data, k_val) {
  kknn_model = kknn(formula = pH ~ .,
                    train = train_data, test = test_data,
                    k = k_val, kernel = "rectangular")
  predicts_kknn = predict(kknn_model)
  mse = mean((test_data$pH - predicts_kknn)^2)
  print(mse)
  return(mse)
}

k_values = c(1, 10, 50)
mse_original = data.frame("train_original" = 0, "validation_original" = 0, "test_original" = 0)
mse_pca = data.frame("train_pca" = 0, "validation_pca" = 0, "test_pca" = 0)
index = c(0)

for(k in k_values) {
  index = index + 1
  mse_original[index, 1] <- kknn_mse(data_train, data_train, k)
  mse_original[index, 2] <- kknn_mse(data_train, data_validation, k)
  mse_original[index, 3] <- kknn_mse(data_train, data_test, k)
  mse_pca[index, 1] <- kknn_mse(combined_train, combined_train, k)
  mse_pca[index, 2] <- kknn_mse(combined_train, combined_validation, k)
  mse_pca[index, 3] <- kknn_mse(combined_train, combined_test, k)
}
mse_original
mse_pca
# > mse_original
# train_original validation_original test_original
# 1 0.0000000 0.4367774 0.4813599
# 2 0.2479928 0.3214056 0.3370375
# 3 0.3852931 0.4285126 0.4746325
# > mse_pca
# train_pca validation_pca test_pca
# 1 0.0000000 0.8685921 0.8221548
# 2 0.4040912 0.4754348 0.5049334
# 3 0.4893246 0.5297236 0.5755813

# Answer:
# 1. Which K leads to the best model from the original data?
# k = 10 leads to the best model because the mean squared error is the lowest

# 2. Which K leads to the best model from the PCA dataset?
# K = 10 leads also to the best model from the PCA dataset with the lowest validation error.

# 3. Which dataset gives the best prediction?
# The original dataset

# 4. Why might one need to check the test error of the best model?
# We select the model based on training, validation error. But to estimate its generalization ability on unseen
# data we check the test error. The test set gives an unbiased estimate how the model will perform in practise.

# 5. Why does K=1 result in zero training MSE?
# When K = 1 each point in the training set is its own nearest neighbor, so the prediction is exactly the same as the true value.
# This makes the training MSE zero but it usually leads to overfitting and poor generalization.

# 6. What target distribution do we implicitly assume by using MSE?
# By using MSE as a cost function, we implicitly assume that the target variable is normally distributed 
# around the regression function with constant variance. MSE corresponds to the maximum likelihood estimator
# under gaussian noise assumptions.


############### Task 3 ###############
# Original data without pH
original_features = select(data_frame, -pH, -LakeID)
original_scaled = predict(preProcess(original_features), original_features)

# PCA data (already only two PCs + pH); exclude pH
pca_features = select(data_combined, -pH)
pca_scaled = predict(preProcess(pca_features), pca_features)

euclidean_ratio <- function(data) {
  d <- as.matrix(dist(data), method = "euclidean")
  d1 <- d[1, -1] # selects all distances from observation 1 to every other observation (exclude itself).
  min(d1) / max(d1) # Ratiop nearest / farthest distance from observation 1
}

ratio_original = euclidean_ratio(original_scaled)
ratio_pca = euclidean_ratio(pca_scaled)
ratio_original
ratio_pca
# Comment:
# Higher ratio (closer to 1) in original high-dimensional space shows distance concentration.
# Lower ratio after PCA (2D) shows better separation of near vs far neighbors.
