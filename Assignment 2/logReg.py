import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import logging
from helper import timing_decorator

logging.basicConfig(filename='logReg_model.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Used data that was not scaled")

@timing_decorator
def logReg(df, test=None, save_df=False, output=False):
    X = df.drop(columns=['srch_id', 'booking_bool', 'click_bool', 'position'])
    X.columns
    y = df['booking_bool']
    
    # Fit the logistic regression model
    logit_model = LogisticRegression(max_iter=10_000, random_state=42)
    result = logit_model.fit(X, y)
    print(result.summary())

    if test is not None:
        # Predict on the test set
        X_test = test.drop(columns=['srch_id'])
        fitted_results = result.predict(logit_model.add_constant(X_test))
        # Rank the results
        ranked_results = np.argsort(-fitted_results)
        sorted_test = test.iloc[ranked_results]
        print(sorted_test.head(10))
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        fitted_results = result.predict(logit_model.add_constant(X_test))
        
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
        submission_df = pd.DataFrame({'srch_id': sorted_test['srch_id'], 'prop_id': sorted_test['prop_id']})
        submission_df.to_csv("submission.csv", index=False)
        print(submission_df.shape[0])
    
    if output:
        return fitted_results, result

if __name__ == "__main__":
    df = pd.read_csv('data\\train_cleaned.csv', nrows=200_000)
    logReg(df)
