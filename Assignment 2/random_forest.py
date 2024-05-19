""" Random Forest Classifier """

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


PATH_TRAIN = "/home/david/Studies/Data Mining Techniques/repository/Assignment 2/data/train_cleaned_100_000.csv"
PATH_TEST = "/home/david/Studies/Data Mining Techniques/repository/Assignment 2/data/test_cleaned_100_000.csv"
PATH_EXPORT = "/home/david/Studies/Data Mining Techniques/repository/Assignment 2/pending_submissions/random_forest.csv"

# Read the CSV files
df_train = pd.read_csv(PATH_TRAIN)
df_test = pd.read_csv(PATH_TEST)
print("Data read successfully")

# Train test split
X_train = df_train.drop(columns=['booking_bool', "click_bool", "position"])
y_train = df_train['booking_bool']

# Algorithm
clf = RandomForestClassifier(n_estimators=100, random_state=42)  # TODO - Try different values for n_estimators on the real dataset
clf.fit(X_train, y_train)

# Rank & sort the results
y_pred = clf.predict(df_test)
# np.argmax(y_pred)  # NOTE - I don't think this line is doing anything
ranked_results = np.argsort(-y_pred)
sorted_test = df_test.iloc[ranked_results]
print(sorted_test.head(10))

# Create submission dataframe
submission_df = pd.DataFrame({'srch_id': sorted_test['srch_id'], 'prop_id': sorted_test['prop_id']})
submission_df.head(10)
submission_df.to_csv(PATH_EXPORT, index=False)
len(submission_df)