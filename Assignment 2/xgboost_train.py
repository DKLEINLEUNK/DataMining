import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


os.chdir("C:\\Users\\Kaleem\\Documents\\Courses\\Data Mining\\DataMining\\Assignment 2")
# Read the CSV file
df_resampled = pd.read_csv('data\\train_cleaned.csv')
df_test = pd.read_csv('data\\test_cleaned.csv')
print("Data read successfully")

############################################################################################
#####     Train-Test Split and Modelling
# Convert the data to DMatrix format for XGBoost
X_train = df_resampled.drop(columns=['booking_bool', "click_bool", "position"])
X_train = X_train.drop(columns=['srch_id', 'prop_id', 'date_time', 'year', 'month', 'day', 'hour'])
y_train = df_resampled['booking_bool']


X = df_resampled.drop(columns=['booking_bool'])
y = df_resampled['booking_bool']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(df_test)

# Set XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'eta': 0.3,
    'eval_metric': 'logloss'
}

# Train the XGBoost model
bst = xgb.train(params, dtrain, num_boost_round=100)
print("Model trained successfully")


############################################################################################
##### Make predictions on the test set and save the results

fitted_results = bst.predict(dtest)
np.argmax(fitted_results)
# fitted_results.head(10)
# Rank the results
ranked_results = np.argsort(-fitted_results)
sorted_test = df_test.iloc[ranked_results]
print(sorted_test.head(10))

# Create submission dataframe
submission_df = pd.DataFrame({'srch_id': sorted_test['srch_id'], 'prop_id': sorted_test['prop_id']})
submission_df.head(10)
submission_df.to_csv("pending_submissions\\xgboost_SMOTE.csv", index=False)
len(submission_df)

### Evaluate the model
# y_pred = (y_pred_prob > 0.5).astype(int)
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)


############################################################################################
### Accuracy measures
# print("Accuracy:", accuracy)
# print("Confusion Matrix:\n", conf_matrix)
# print("Classification Report:\n", class_report)

