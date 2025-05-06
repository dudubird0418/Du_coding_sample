# ---------------------------------------------------------------
# Code Sample for  Postgraduate Research Associate, Organizational Behavior
# Author: Wenlu Du
# Description:
# This code demonstrates my data processing, feature engineering,
# and machine learning capabilities using R in the context of a 
# longitudinal panel study on COVID-19 vaccination behavior.
#
# Core components include:
# - Data wrangling from SPSS (3-wave panel)
# - 5C psychological model variable construction
# - Feature mapping, reverse coding, and scoring
# - Model building: Naive Bayes, Decision Tree, Random Forest, GBM
# - ROC analysis, AUC comparison, Confusion Matrix evaluation
# - Weight of Evidence (WoE) calculation for categorical predictors
#
# This project was part of my RA work at Penn's Social Action Lab 
# as well as machine learning coursework ,
# combining social psychology theory with predictive modeling
# to examine vaccine uptake behavior.
# ---------------------------------------------------------------

# Load necessary libraries
library(foreign)
library(naivebayes)
library(caret)
library(pROC)
library(rpart)
library(randomForest)
library(gbm)
library(dplyr)
library(tidyr)
library(knitr)
library(kableExtra)
library(purrr)
library(Hmisc)

# Load the data
data_wave_1_2_3_combined <- read.spss("/Users/duwenlu/Academic/SAL/SSRSLongitudinalVaccinationStudy_from Science Communication/V4049.V4050.W4012_APPC COVID-19 Longitudinal Panel Study_W1 + W2 + W3 Combined_WEIGHT OFF_071223_Confidential.sav", 
                                      to.data.frame = TRUE)

# Define mappings for the 5C model
predictors_mapping <- list(
  Q14 = "confidence", Q15 = "confidence", Q16 = "confidence",
  Q7 = "complacency", Q8 = "complacency", Q9 = "complacency",
  Q48 = "constraints", Q50 = "constraints", Q51 = "constraints", Q52 = "constraints", Q53 = "constraints",
  Q25 = "risk_calculation", Q26 = "risk_calculation", Q27 = "risk_calculation", Q28 = "risk_calculation", Q29 = "risk_calculation",
  Q30 = "collective_responsibility", Q41 = "collective_responsibility", Q42 = "collective_responsibility"
)
outcomes_mapping <- list(
  Q1 = "behavior_vaccine", Q2 = "behavior_booster",
  Q12 = "intention_vaccine", Q13 = "intention_booster"
)
demographic_vars <- c("PGENDER", "PEDUC", "PRACE", "PINCOME", "PSTATE", "PAGE1")

# Function to rename variables
rename_variable <- function(question, wave, mapping) {
  base_name <- mapping[[question]]
  if (!is.null(base_name)) {
    return(paste0(base_name, "_w", wave, "_", question))
  } else {
    return(paste0(question, "_w", wave))
  }
}

# Adjust the column selection logic
columns_to_extract <- c()
renamed_columns <- c()
for (wave in 1:3) {
  for (question in names(predictors_mapping)) {
    column <- paste0(question, "_w", wave)
    columns_to_extract <- c(columns_to_extract, column)
    renamed_columns <- c(renamed_columns, rename_variable(question, wave, predictors_mapping))
  }
  for (question in names(outcomes_mapping)) {
    column <- paste0(question, "_w", wave)
    columns_to_extract <- c(columns_to_extract, column)
    renamed_columns <- c(renamed_columns, rename_variable(question, wave, outcomes_mapping))
  }
  for (demographic in demographic_vars) {
    column <- paste0(demographic, "_w", wave)
    columns_to_extract <- c(columns_to_extract, column)
    renamed_columns <- c(renamed_columns, paste0(demographic, "_w", wave))
  }
}

# Extract the data
extracted_data <- data_wave_1_2_3_combined[, columns_to_extract, drop = FALSE]
names(extracted_data) <- renamed_columns

# Map each response category to numeric scores per question, considering riverse cording
question_score_mapping <- list(
  Q14 = c("Very negative" = 1, "Somewhat negative" = 2, "Neutral" = 3, "Somewhat positive" = 4, "Very positive" = 5),
  Q15 = c("Not at all risky" = 5, "A little risky" = 4, "Somewhat risky" = 3, "Very risky" = 2, "Extremely risky" = 1),
  Q16 = c("Not at all effective" = 1, "A little effective" = 2, "Somewhat effective" = 3, "Very effective" = 4, "Extremely effective" = 5),
  Q7 = c("Not at all likely" = 4, "Not too likely" = 3, "Somewhat likely" = 2, "Very likely" = 1),
  Q8 = c("Not at all severe" = 4, "Not too severe" = 3, "Somewhat severe" = 2, "Very severe" = 1),
  Q9 = c("Less severe" = 3, "Just as severe" = 2, "More severe" = 1),
  Q48 = c("Extremely easy" = 1, "Somewhat easy" = 2, "Neither easy nor difficult" = 3, "Somewhat difficult" = 4, "Extremely difficult" = 5),
  Q50 = c("Extremely easy" = 1, "Somewhat easy" = 2, "Neither easy nor difficult" = 3, "Somewhat difficult" = 4, "Extremely difficult" = 5),
  Q51 = c("Extremely easy" = 1, "Somewhat easy" = 2, "Neither easy nor difficult" = 3, "Somewhat difficult" = 4, "Extremely difficult" = 5),
  Q52 = c("Extremely easy" = 1, "Somewhat easy" = 2, "Neither easy nor difficult" = 3, "Somewhat difficult" = 4, "Extremely difficult" = 5),
  Q53 = c("Extremely easy" = 1, "Somewhat easy" = 2, "Neither easy nor difficult" = 3, "Somewhat difficult" = 4, "Extremely difficult" = 5),
  Q25 = c("Not at all likely" = 5, "A little likely" = 4, "Somewhat likely" = 3, "Very likely" = 2, "Extremely likely" = 1),
  Q26 = c("Not at all likely" = 5, "A little likely" = 4, "Somewhat likely" = 3, "Very likely" = 2, "Extremely likely" = 1),
  Q27 = c("Not at all likely" = 5, "A little likely" = 4, "Somewhat likely" = 3, "Very likely" = 2, "Extremely likely" = 1),
  Q28 = c("Not at all likely" = 5, "A little likely" = 4, "Somewhat likely" = 3, "Very likely" = 2, "Extremely likely" = 1),
  Q29 = c("Not at all likely" = 5, "A little likely" = 4, "Somewhat likely" = 3, "Very likely" = 2, "Extremely likely" = 1),
  Q30 = c("Not at all likely" = 1, "A little likely" = 2, "Somewhat likely" = 3, "Very likely" = 4, "Extremely likely" = 5),
  Q41 = c("Not at all worried" = 1, "A little worried" = 2, "Not too worried" = 3, "Somewhat worried" = 4, "Very worried" = 5),
  Q42 = c("Not at all important" = 1, "A little important" = 2, "Somewhat important" = 3, "Very important" = 4, "Extremely important" = 5)
)

# Function to map responses to numerical scores
map_scores <- function(question_col, mapping) {
  return(as.numeric(factor(question_col, levels = names(mapping), labels = mapping)))
}

# Apply the scoring to the extracted data
scored_data <- extracted_data
for (question in names(question_score_mapping)) {
  cols <- grep(question, names(scored_data), value = TRUE)
  for (col in cols) {
    scored_data[[col]] <- map_scores(scored_data[[col]], question_score_mapping[[question]])
  }
}

# Function to compute average scores by wave
compute_average_per_wave <- function(cols) {
  numeric_data <- scored_data[, cols, drop = FALSE]
  wave1 <- rowMeans(numeric_data[grep("_w1_", names(numeric_data))], na.rm = TRUE)
  wave2 <- rowMeans(numeric_data[grep("_w2_", names(numeric_data))], na.rm = TRUE)
  wave3 <- rowMeans(numeric_data[grep("_w3_", names(numeric_data))], na.rm = TRUE)
  return(data.frame(wave1, wave2, wave3))
}

# Compute the average scores per dimension, excluding intention and behavior variables
dimension_columns <- list(
  "confidence" = grep("confidence", names(scored_data), value = TRUE),
  "complacency" = grep("complacency", names(scored_data), value = TRUE),
  "constraints" = grep("constraints", names(scored_data), value = TRUE),
  "risk_calculation" = grep("risk_calculation", names(scored_data), value = TRUE),
  "collective_responsibility" = grep("collective_responsibility", names(scored_data), value = TRUE)
)

# Compute average scores per wave for each dimension
dimension_averages <- lapply(dimension_columns, compute_average_per_wave)

# Combine the results into a single data frame and rename columns
average_scores <- do.call(cbind, dimension_averages)
colnames(average_scores) <- unlist(lapply(names(dimension_averages), function(dim) {
  paste(dim, c("w1", "w2", "w3"), sep = "_")
}))

head(average_scores)

# Extract demographic and outcome columns explicitly
demographic_cols <- paste(demographic_vars, collapse = "|")
demographic_data <- extracted_data[, grep(demographic_cols, names(extracted_data), value = TRUE)]

outcome_cols <- paste(names(outcomes_mapping), collapse = "|")
outcome_data <- extracted_data[, grep(outcome_cols, names(extracted_data), value = TRUE)]

# Combine average_scores with demographic and outcome data
final_combined_data <- cbind(demographic_data, average_scores, outcome_data)

# Check the final combined data
head(final_combined_data)


# Mapping of old demographic names to new descriptive names
demographic_rename_map <- list(
  "PAGE1" = "age",
  "PINCOME" = "income",
  "PGENDER" = "gender",
  "PEDUC" = "education",
  "PRACE" = "race",
  "PSTATE" = "state"
)

# Function to rename demographic columns based on the rename map
rename_demographic_columns <- function(colnames) {
  new_names <- sapply(colnames, function(name) {
    name_split <- strsplit(name, "_")[[1]]
    base_name <- name_split[1]
    wave_suffix <- ifelse(length(name_split) > 1, name_split[2], "")
    new_base_name <- ifelse(base_name %in% names(demographic_rename_map), 
                            demographic_rename_map[[base_name]], 
                            base_name)
    paste0(new_base_name, if (wave_suffix != "") paste0("_", wave_suffix) else "")
  })
  return(new_names)
}

# Apply the renaming function to demographic columns in the final data
demographic_col_indices <- grep(paste(demographic_vars, collapse = "|"), names(final_combined_data))
colnames(final_combined_data)[demographic_col_indices] <- rename_demographic_columns(colnames(final_combined_data)[demographic_col_indices])

# Check the updated column names
colnames(final_combined_data)

# Define target and predictors for wave 3
target <- "binary_behavior_vaccine_w3"
predictors_5c <- c("confidence_w1", "complacency_w1", "constraints_w1", "risk_calculation_w1", "collective_responsibility_w1")
demographic_predictors <- c("gender_w1", "education_w1", "race_w1", "income_w1", "state_w1", "age_w1")
predictors <- c(predictors_5c, demographic_predictors)

# List of columns needed for analysis
columns_needed <- c(predictors, target)

# Convert vaccine response to binary
final_combined_data$binary_behavior_vaccine_w3 <- ifelse(
  final_combined_data$behavior_vaccine_w3_Q1 %in% c(
    "Yes, two doses of the Moderna or Pfizer vaccine",
    "Yes, only one dose of the Johnson and Johnson vaccine",
    "Yes, only one dose of the Moderna or Pfizer vaccine"
  ), 1, 0)

# Subset the data
final_combined_data_subset <- final_combined_data[columns_needed]

############ Descriptive Analysis #################################
# Summary Statistics
summary_stats <- summary(final_combined_data_subset)
print(summary_stats)

####################### Naive bayes ###############################
# Ensure predictors are factors or numeric for Naive Bayes model
categorical_cols <- c("gender_w1", "education_w1", "race_w1", "state_w1", "income_w1")
final_combined_data[categorical_cols] <- lapply(final_combined_data[categorical_cols], as.factor)
numeric_cols <- setdiff(predictors, categorical_cols)
final_combined_data[numeric_cols] <- lapply(final_combined_data[numeric_cols], as.numeric)

############According to the distribution plots we better standardize the numberic variables
final_combined_data[numeric_cols] <- scale(final_combined_data[numeric_cols])

# Split data into training and validation sets (80% training, 20% validation)
set.seed(20240506)

# Define the predictors and target correctly
predictors_5c <- c("confidence_w1", "complacency_w1", "constraints_w1", "risk_calculation_w1", "collective_responsibility_w1")
demographic_predictors <- c("gender_w1", "education_w1", "race_w1", "income_w1", "age_w1")  # Removed `state_w1` if it's causing inconsistency
predictors <- c(predictors_5c, demographic_predictors)

target <- "binary_behavior_vaccine_w3"

# Prepare the final dataset
columns_needed <- c(predictors, target)
final_combined_data_subset <- final_combined_data[columns_needed]

# Split into training and validation sets
train_index <- createDataPartition(final_combined_data_subset[[target]], p = 0.8, list = FALSE, times = 1)
train_set <- final_combined_data_subset[train_index, ]
validation_set <- final_combined_data_subset[-train_index, ]
# Remove rows with missing values
train_set <- na.omit(train_set)

# Ensure the target variable is a factor
train_set[[target]] <- factor(train_set[[target]], levels = c(0, 1), labels = c("No", "Yes"))
validation_set[[target]] <- factor(validation_set[[target]], levels = c(0, 1), labels = c("No", "Yes"))

# Train the Naive Bayes model
nb_model <- naive_bayes(as.formula(paste(target, "~", paste(predictors, collapse = "+"))), data = train_set, laplace = 1)

# Make predictions on the validation set
nb_predictions <- predict(nb_model, newdata = validation_set[, predictors], type = "prob")[, "Yes"]

# Convert probabilities to binary predictions
nb_binary_preds <- ifelse(nb_predictions > 0.5, "Yes", "No")
nb_binary_preds <- factor(nb_binary_preds, levels = levels(validation_set[[target]]))

# Evaluate Naive Bayes Performance
nb_roc <- roc(validation_set[[target]], nb_predictions)
nb_auc <- auc(nb_roc)
cat("Naive Bayes AUC: ", nb_auc, "\n")

# Plot ROC Curve
plot(nb_roc, main = "ROC Curve for Naive Bayes Model", col = "blue")
abline(a = 0, b = 1, col = "red", lty = 2)

# Confusion Matrix
nb_conf_matrix <- confusionMatrix(nb_binary_preds, validation_set[[target]])
print(nb_conf_matrix)


# Train Random Forest model
rf_model <- randomForest(
  as.formula(paste(target, "~", paste(predictors, collapse = "+"))),
  data = train_set,
  ntree = 500,
  importance = TRUE
)

# Make predictions on the validation set
rf_predictions <- predict(rf_model, newdata = validation_set, type = "prob")[, "Yes"]

# Convert probabilities to binary predictions
rf_binary_preds <- ifelse(rf_predictions > 0.5, "Yes", "No")
rf_binary_preds <- factor(rf_binary_preds, levels = levels(validation_set[[target]]))

# Plot ROC Curve for Random Forest
plot(rf_roc, main = "ROC Curve for Random Forest Model", col = "purple")
abline(a = 0, b = 1, col = "red", lty = 2)

# Evaluate Random Forest Performance
rf_roc <- roc(validation_set[[target]], rf_predictions)
rf_auc <- auc(rf_roc)
cat("Random Forest AUC: ", rf_auc, "\n")

# Confusion Matrix
rf_conf_matrix <- confusionMatrix(rf_binary_preds, validation_set[[target]])
print(rf_conf_matrix)


################Decision Trees###############
# Define control parameters
control_params <- rpart.control(
  minsplit = 20,
  minbucket = 7,
  cp = 0.01,
  maxdepth = 5,
  xval = 10
)

# Train the Decision Tree model with control parameters
dt_model <- rpart(
  formula = as.formula(paste(target, "~", paste(predictors, collapse = "+"))),
  data = train_set, 
  method = "class",
  control = control_params
)

# Predictions on the validation data
dt_predictions <- predict(dt_model, newdata = validation_set, type = "prob")[, 2]

# Convert probabilities to binary predictions
dt_binary_preds <- ifelse(dt_predictions > 0.5, 1, 0)
dt_binary_preds <- factor(dt_binary_preds, levels = c(0, 1))

# Evaluate Decision Tree Performance
dt_roc <- roc(validation_set[[target]], dt_predictions)
dt_auc <- auc(dt_roc)
cat("Decision Tree AUC: ", dt_auc, "\n")

# Plot ROC Curve
plot(dt_roc, main = "ROC Curve for Decision Tree Model", col = "green")
abline(a = 0, b = 1, col = "red", lty = 2)

# Convert target values to 0 and 1
actuals <- as.numeric(validation_set[[target]] == "Yes")
actuals <- factor(actuals, levels = c(0, 1))

# Ensure predicted values are factors with levels 0 and 1
dt_binary_preds <- factor(dt_binary_preds, levels = c(0, 1))

# Confusion Matrix
dt_conf_matrix <- confusionMatrix(dt_binary_preds, actuals)
print(dt_conf_matrix)

##########################
# Convert the target variable to binary (0 and 1) if not already
train_set[[target]] <- as.numeric(train_set[[target]] == "Yes")
validation_set[[target]] <- as.numeric(validation_set[[target]] == "Yes")

# Verify the encoding
print(unique(train_set[[target]]))
print(unique(validation_set[[target]]))

# Train the boosting model
boosting_model <- gbm(
  formula = as.formula(paste(target, "~", paste(predictors, collapse = "+"))),
  data = train_set,
  distribution = "bernoulli",
  n.trees = 3000,
  interaction.depth = 3,
  shrinkage = 0.01,
  n.minobsinnode = 10,
  cv.folds = 5,
  verbose = FALSE
)

# Optimal number of trees
best_trees <- gbm.perf(boosting_model, method = "cv")

# Make predictions with the boosting model
boosting_predictions <- predict(boosting_model, newdata = validation_set, n.trees = best_trees, type = "response")

# Convert probabilities to binary predictions
boosting_binary_preds <- ifelse(boosting_predictions > 0.5, 1, 0)
boosting_binary_preds <- factor(boosting_binary_preds, levels = c(0, 1))

# Evaluate the boosting model
boosting_roc <- roc(validation_set[[target]], boosting_predictions)
boosting_auc <- auc(boosting_roc)
cat("Boosting Model AUC: ", boosting_auc, "\n")

# Plot ROC Curve
plot(boosting_roc, main = "ROC Curve for Boosting Model", col = "blue")
abline(a = 0, b = 1, col = "red", lty = 2)

# Confusion Matrix
boosting_conf_matrix <- confusionMatrix(factor(boosting_binary_preds), factor(validation_set[[target]]))
print(boosting_conf_matrix)

##################Direct Comparison###############
##################Indices###########################
# Prepare data for model comparison
model_names <- c("Naive Bayes", "Random Forest", "Decision Tree", "Boosting")
conf_matrices <- list(nb_conf_matrix, rf_conf_matrix, dt_conf_matrix, boosting_conf_matrix)
auc_values <- c(nb_auc, rf_auc, dt_auc, boosting_auc)

# Extract metrics from confusion matrices
get_metrics <- function(conf_matrix) {
  c(Accuracy = conf_matrix$overall["Accuracy"],
    Sensitivity = conf_matrix$byClass["Sensitivity"],
    Specificity = conf_matrix$byClass["Specificity"])
}

metrics <- sapply(conf_matrices, get_metrics)
metrics <- t(metrics)
rownames(metrics) <- model_names

# Add AUC values
metrics <- cbind(metrics, AUC = auc_values)

# Create a formatted table using kable
kable(metrics, format = "html", digits = 3, align = "c", col.names = c("Accuracy", "Sensitivity", "Specificity", "AUC")) %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), full_width = F) %>%
  add_header_above(c(" " = 1, "Model Comparison" = 4)) %>%
  column_spec(1, bold = TRUE)

###################ROC plots######################

# Plotting the ROC curves together
plot(nb_roc, col = "blue", main = "Comparison of ROC Curves for Different Models", lwd = 2, xlab = "False Positive Rate", ylab = "True Positive Rate")
lines(rf_roc, col = "purple", lwd = 2)
lines(dt_roc, col = "green", lwd = 2)
lines(boosting_roc, col = "red", lwd = 2)
abline(a = 0, b = 1, col = "gray", lty = 2)

# Adding a legend
legend("bottomright", legend = c("Naive Bayes", "Random Forest", "Decision Tree", "Boosting"),
       col = c("blue", "purple", "green", "red"), lwd = 2)


################# WoE Sheet ######################

# Function to compute weights of evidence
compute_woe <- function(x, y) {
  tbl <- table(y, x, useNA = "no")  # Ensure no NA values are used
  probs <- prop.table(tbl, 1)  # Normalize over rows
  woe <- log(probs[2, ] / probs[1, ])  # Calculate WoE
  return(woe)
}

# Function to compute WoE for a single categorical predictor
compute_woe_for_predictor <- function(data, predictor, target) {
  if (is.factor(data[[predictor]])) {  # Only calculate WoE for factor variables
    woes <- compute_woe(data[[predictor]], data[[target]])
    data.frame(
      predictor = predictor,
      level = names(woes),
      woe = woes,
      stringsAsFactors = FALSE
    )
  } else {
    data.frame()  # Return empty data frame for non-factor variables
  }
}

# Ensure the target variable is binary and factor type
final_combined_data_subset$binary_behavior_vaccine_w3 <- factor(final_combined_data_subset$binary_behavior_vaccine_w3, levels = c(0, 1), labels = c("No", "Yes"))

# Define the categorical predictors
categorical_predictors <- c("gender_w1", "education_w1", "race_w1", "income_w1", "state_w1")

# Compute WoE for each categorical predictor and combine results into a data frame
woe_list <- lapply(categorical_predictors, function(pred) compute_woe_for_predictor(final_combined_data_subset, pred, target))
modNB <- bind_rows(woe_list)

# Create the WoE Table
woe_table <- modNB %>%
  mutate(woe = round(woe, 2)) %>%  # Adjust rounding if necessary
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover"))

# Display the whole WoE Table
print(woe_table)
