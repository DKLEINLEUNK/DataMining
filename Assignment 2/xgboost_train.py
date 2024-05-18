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
from datetime import datetime
import time
import dask.dataframe as dd
import logging
from helper import timing_decorator

# Set up logging
logging.basicConfig(filename='xgboost_model.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Used data that was not scaled")

@timing_decorator
def xg_boost(data, test=None, save_df=False, output=False):
    """
    Train an XGBoost model using the given data and make predictions on the test set.

    Parameters:
    - data: DataFrame, the training data.
    - test: DataFrame, the test data. If provided, predictions will be made on this test set.
    - save_df: bool, whether to save the submission dataframe.

    Returns:
    - fitted_results: Series, the predictions made on the test set.
    - 
    """
    # Set XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'eta': 0.3,
        'eval_metric': 'logloss'
    }

    if test is not None:
        X_train = data.drop(columns=["srch_id", 'booking_bool', "click_bool", "position"])
        X_test = test.drop(columns=["srch_id"])
        predictors = X_train.columns
        predictors_test = X_test.columns
        common_cols = list(set(predictors).intersection(predictors_test))
        y_train = data['booking_bool']
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, enable_categorical=True)

        # Train the XGBoost model
        bst = xgb.train(params, dtrain, num_boost_round=100)
        print("Model trained successfully")
        ##### Make predictions on the test set and save the results

        fitted_results = bst.predict(dtest)
        np.argmax(fitted_results)
        # Rank the results
        ranked_results = np.argsort(-fitted_results)
            
    else:
        X = data.drop(columns=["srch_id", 'booking_bool', "click_bool", "position"])
        y = data['booking_bool']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
        
        # Train the XGBoost model
        bst = xgb.train(params, dtrain, num_boost_round=100)
        print("Model trained successfully")
        ##### Make predictions on the test set and save the results

        fitted_results = bst.predict(dtest)
        np.argmax(fitted_results)
        # Rank the results
        ranked_results = np.argsort(-fitted_results)
        
        # Evaluate the model
        y_pred = (fitted_results > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Confusion Matrix:\n", conf_matrix)
        print("Classification Report:\n", class_report)
        
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")
        logging.info(f"Classification Report:\n{class_report}")

        
    
    if save_df:
        # Create submission dataframe
        sorted_test = test.iloc[ranked_results]
        print(sorted_test.head(10))
        submission_df = pd.DataFrame({'srch_id': sorted_test['srch_id'], 'prop_id': sorted_test['prop_id']})
        submission_df.head(10)
        submission_df.to_csv(f"pending_submissions\\xgboost_scaled_{save_df}.csv", index=False)
    
    
    if output:
        return fitted_results, bst
    
if __name__ == "__main__":
    os.chdir("C:\\Users\\Kaleem\\Documents\\Courses\\Data Mining\\DataMining\\Assignment 2")
    # Load the data
    data = pd.read_csv("data\\train_cleaned.csv", nrows=100_000)
    # xg_boost(data)
    test = pd.read_csv("data\\test_cleaned.csv", nrows=100_000)
    xg_boost(data, test, save_df='updated')
    new_df = pd.read_csv("pending_submissions\\xgboost_scaled_updated.csv")
    if len(new_df) == 4959183:
        print("Submission file matches requirements")

############################################################################################
### Accuracy measures
# print("Accuracy:", accuracy)
# print("Confusion Matrix:\n", conf_matrix)
# print("Classification Report:\n", class_report)

