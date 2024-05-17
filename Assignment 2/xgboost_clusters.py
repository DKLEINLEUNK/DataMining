from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import xgboost as xgb
import numpy as np
import os

os.chdir("C:\\Users\\Kaleem\\Documents\\Courses\\Data Mining\\DataMining\\Assignment 2")
# Read the CSV files
train_df = pd.read_csv('data\\train_clusters.csv')
test_df = pd.read_csv('data\\test_clusters.csv')
print("Data read successfully")

# Split the data by clusters
clusters = train_df['Cluster'].unique()
train_clustered_data = {cluster: train_df[train_df['Cluster'] == cluster] for cluster in clusters}
test_clustered_data = {cluster: test_df[test_df['Cluster'] == cluster] for cluster in clusters}

# Display the sizes of each cluster
for cluster, data in train_clustered_data.items():
    print(f"Cluster {cluster}: {data.shape[0]} samples")

for cluster, data in test_clustered_data.items():
    print(f"Cluster {cluster}: {data.shape[0]} samples")


cluster=0
data = train_clustered_data[cluster]

# Set XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'eta': 0.3,
    'eval_metric': 'logloss'
}


# Train and evaluate a model for each cluster
for cluster, data in train_clustered_data.items():
    # Split the cluster data into features (X) and target (y)
    X_train = data.drop(columns=['booking_bool', 'Cluster', "target"])
    y_train = data['booking_bool']
    X_test = test_df[test_df['Cluster'] == cluster]
    X_test = X_test.drop(columns=['Cluster'])
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    
    print("Data read successfully for cluster", cluster,"...")
    
    bst = xgb.train(params, dtrain, num_boost_round=100)
    print("Model trained successfully for cluster", cluster,"...")

    print(X_train.columns.isin(X_test.columns))
    # Make predictions on the test set
    y_pred = bst.predict(dtest)
    
    fitted_results = bst.predict(dtest)
    X_test["prediction"] = fitted_results
    # fitted_results.head(10)
    # Rank the results
    ranked_results = np.argsort(-fitted_results)
    sorted_test = X_test.iloc[ranked_results]
    
    submission_df = pd.DataFrame({'srch_id': sorted_test['srch_id'], 'prop_id': sorted_test['prop_id'], "probabililty": sorted_test['prediction']})
    submission_df.to_csv(f"pending_submissions\\xgboost_cluster_{cluster}.csv", index=False)
    print("Submission file created successfully for cluster", cluster,"...")


## Combine results from all clusters
submission_files = [f"pending_submissions\\xgboost_cluster_{cluster}.csv" for cluster in clusters]
submission_dfs = [pd.read_csv(file) for file in submission_files]
submission_df = pd.concat(submission_dfs)
submission_df.sort_values(by=['probabililty'], ascending=False, inplace=True)
submission_final = submission_df.drop(columns=['probabililty'])
submission_final.to_csv("pending_submissions\\xgboost_clusters.csv", index=False)


submission_df_final = pd.read_csv("pending_submissions\\xgboost_clusters.csv")

len(submission_df_final)