import numpy as np
import pandas as pd
from cuml.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import logging
from cuml.metrics import accuracy_score, confusion_matrix
from helper import timing_decorator


from cuml.datasets.classification import make_classification
from cuml.model_selection import train_test_split
from cuml.ensemble import RandomForestClassifier as cuRF
from sklearn.metrics import accuracy_score
# import dask.dataframe as dd


# Set up logging
logging.basicConfig(filename='random_forrest.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Random forrest with all engineered features	")


@timing_decorator
def randomForrest(df, test=None, save_df=False, output=False):
    # X = df.drop(columns=['srch_id', 'booking_bool', 'click_bool', 'position'])
    # y = df['booking_bool']

    # Initialize the Random Forest Regressor
    rf_regressor = cuRF(max_depth=10,
                 n_estimators=25,
                 random_state=42)


    if test is not None:
        X_train = df.drop(columns=["srch_id", 'booking_bool', "click_bool", "position"])
        y_train = df['booking_bool']

        X_test = test.drop(columns=["srch_id"])
        predictors = X_train.columns
        predictors_test = X_test.columns
        common_cols = list(set(predictors).intersection(predictors_test))

        # Train the model
        rf_regressor.fit(X_train, y_train)

        # Make predictions
        fitted_results = rf_regressor.predict(X_test)

        # Rank the results
        ranked_results = np.argsort(-fitted_results)
        sorted_test = test.iloc[ranked_results]
        print(sorted_test.head(10))
    else:
        X = df.drop(columns=["srch_id", 'booking_bool', "click_bool", "position"])
        y = df["booking_bool"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        rf_regressor.fit(X_train, y_train)

        fitted_results = rf_regressor.predict(X_test)

        y_pred = (fitted_results > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        # class_report = classification_report(y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Confusion Matrix:\n", conf_matrix)
        # print("Classification Report:\n", class_report)

        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")
        # logging.info(f"Classification Report:\n{class_report}")

    if save_df:
        # Create submission dataframe
        submission_df = pd.DataFrame({'srch_id': sorted_test['srch_id'], 'prop_id': sorted_test['prop_id']})
        submission_df.to_csv("random_forrest.csv", index=False)
        print(submission_df.shape[0])

    if output:
        return fitted_results, fitted_results  # Initialize the Random Forest Regressor


if __name__ == "__main__":
    import cudf
    import dask_cudf
    from datetime import datetime

    # logging.info("Random forrest with all engineered features (using CUDA")
    start_time = datetime.now()
    # %load_ext cudf.pandas
    # import os
    #    os.chdir("C:\\Users\\Kaleem\\Documents\\Courses\\Data Mining\\DataMining\\Assignment 2")
    # Load the data

    PATH_train = "/windows/Users/Kaleem/Documents/Courses/Data Mining/DataMining/Assignment 2/data/train_cleaned.csv"
    # df = cudf.read_csv(PATH_train)
    df = dask_cudf.from_cudf(cudf.read_csv(PATH_train, nrows=100_000), npartitions=1000).compute()

    # test = pd.read_csv("data\\test_cleaned.csv")
    randomForrest(df, output=False)

    end_time = datetime.now()

    print('Duration: {}'.format(end_time - start_time))
    logging.info('Duration: {}'.format(end_time - start_time))