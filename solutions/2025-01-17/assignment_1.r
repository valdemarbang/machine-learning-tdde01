setwd("~/Repos/machine-learning-tdde01/solutions/2025-01-17")

### library ###
library(dplyr)
library(caret)
library(kknn)

### task 1.1 ###
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
pca_raw <- princomp(df_no_ph_cc, cor = FALSE)  # 2) Run PCA on raw data (no scaling)
summary(pca_raw) # 3) Summary of variance explained
loadings(pca_raw) # 4) Inspect loadings (in original units)
scores_raw <- pca_raw$scores[, 1:2]  # # 5) Get PC scores first two PCs
# 6) Approximate Cond from first two PCs directly (no scaling needed)
l_cond <- pca_raw$loadings["Cond", 1:2]
Cond_hat_raw <- scores_raw %*% l_cond

# 7) View first 10 comparisons
head(cbind(Cond_true = df_no_ph_cc$Cond,
           Cond_hat  = Cond_hat_raw), 10)
