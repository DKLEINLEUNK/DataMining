require(MASS)
require(lme4)
require(dplyr)
# setwd("/windows/Users/Kaleem/Documents/Courses/Data Mining/DataMining/Assignment 2")
setwd("C://Users//Kaleem//Documents//Courses//Data Mining//DataMining//Assignment 2")

############################
## Complete set
df <- read.csv("data/train.csv")
df$booking_bool <- as.factor(df$booking_bool)
df$srch_id <- as.factor(df$srch_id)
df$promotion_flag <- as.factor(df$promotion_flag)
df$random_bool <- as.factor(df$random_bool)
df$prop_id <- as.factor(df$prop_id)
# Scaling and fixing variable type
scaled_df <- df %>%
  mutate_if(is.numeric, scale)
scaled_df <- scaled_df %>%
  mutate_if(is.matrix, as.numeric)
clean_df <- na.omit(scaled_df)
nrow(clean_df)

#########################################
# Dividing the dataset to test
df_train <- df[sample(nrow(df), 1000), ]
head(df_train, 10)
# df_test <- tail(clean_df, 500000)
# head(df_test, 10)
# head(df$random_bool)

# forward_model <- glm(booking_bool ~ ., data=df)
# forward_model <- step(forward_model, direction = "forward", scope = formula(~ .))

# Define control settings for the optimizer
# control_settings <- glmerControl(optimizer = "bobyqa",
#                                  optCtrl = list(maxfun = 1e5),  # increase max function evaluations
#                                  calc.derivs = TRUE,
#                                  check.conv.grad = .makeCC("warning", tol = 1e-3, relTol = NULL))



##########################################################
# Factors 
# clean_df$booking_bool <- as.factor(clean_df$booking_bool)
# clean_df$srch_id <- as.factor(clean_df$srch_id)
# clean_df$promotion_flag <- as.factor(clean_df$promotion_flag)
# clean_df$random_bool <- as.factor(clean_df$random_bool)


##########################################################
## Actual Test set

test_set <- read.csv("data//test.csv")
# test_set_sub <- select(test_set, c("random_bool", "price_usd", "promotion_flag", "srch_destination_id", "srch_room_count", "srch_length_of_stay", "srch_id", "prop_id")) 
test_final$srch_id <- as.factor(test_final$srch_id)
test_final$promotion_flag <- as.factor(test_final$promotion_flag)
test_final$random_bool <- as.factor(test_final$random_bool)
test_final$prop_id <- as.factor(test_final$prop_id)

scaled_test <- test_final %>%
  mutate_if(is.numeric, scale)
scaled_test <- scaled_test %>%
  mutate_if(is.matrix, as.numeric)
nrow(scaled_test)
# clean_test <- na.omit(scaled_test)
# test_final <- na.omit(test_set)
# 
# class(clean_test$price_usd)
# class(clean_df$price_usd)
##########################################################
# Log-Reg
formula_logreg <- "booking_bool ~ prop_review_score + price_usd + prop_starrating + random_bool + price_usd + promotion_flag + srch_destination_id + srch_room_count + srch_length_of_stay"
book_log <- glm(formula_logreg, family=binomial(link="logit"), data = df) 
# book_log_stepwise <- glm(formula_logreg, family=binomial(link="logit"), data = df) %>%
#   stepAIC(trace = FALSE)
summary(book_log)

# # Mixed Model Fit 1
# book_fit <- glmer(
#   booking_bool ~ random_bool + promotion_flag + srch_destination_id + srch_room_count + srch_length_of_stay + (1 | prop_id),
#                         data = clean_df, 
#                         family = binomial)
# summary(book_fit)

#########################################################
## Predictions
fitted.results <- predict(book_log,newdata=test_set,type='response')
head(fitted.results)
max(fitted.results)

class(fitted.results)

ranked_results <- order(fitted.results, decreasing = T)
sorted_test <- test_set[ranked_results, ]
head(sorted_test, 10)
submission_df <- data.frame(srch_id = sorted_test$srch_id, 
                            prop_id = sorted_test$prop_id)

write.csv(submission_df, "submission.csv", row.names = F)
nrow(submission_df)

# head(ranked_results)
# fitted.results <- ifelse(fitted.results > 0.5,1,0)
# max(fitted.results)
# misClasificError <- mean(fitted.results != df_test$booking_bool)
# print(paste('Accuracy',1-misClasificError))


emm <- emmeans::emmeans(book_fit, ~.)

summary(book_fit)
