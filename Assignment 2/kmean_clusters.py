import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

os.chdir("C:\\Users\\Kaleem\\Documents\\Courses\\Data Mining\\DataMining\\Assignment 2")

df = pd.read_csv('data\\train_cleaned.csv')
df_test = pd.read_csv('data\\test_cleaned.csv')
df.head(10)
print("Data read successfully")

df_predictor = df.drop(columns=['booking_bool', 'click_bool', 'position'])
predictors = df_predictor.columns
df_predictor['booking_bool'] = df['booking_bool']
df_predictor['click_bool'] = df['click_bool']
df_predictor['position'] = df['position']


# Initialize the KMeans algorithm
kmeans = KMeans(n_clusters=5, random_state=42)
# Fit the KMeans algorithm on the data
kmeans.fit(df_predictor[predictors])

# Predict the clusters
train_clusters = kmeans.predict(df_predictor)
test_clusters = kmeans.predict(df_test[predictors])

# Count the number of data points in each cluster
len(train_clusters[train_clusters == 0])
len(train_clusters[train_clusters == 1])
len(train_clusters[train_clusters == 2])
len(train_clusters[train_clusters == 3])
len(train_clusters[train_clusters == 4])

len(test_clusters[test_clusters == 0])
len(test_clusters[test_clusters == 1])
len(test_clusters[test_clusters == 2])
len(test_clusters[test_clusters == 3])
len(test_clusters[test_clusters == 4])




# Add the cluster assignments to the DataFrame
df_predictor['Cluster'] = train_clusters
df_test['Cluster'] = test_clusters

df_predictor.to_csv('data\\train_clusters.csv')
df_test.to_csv('data\\test_clusters.csv')

print("Clusters added successfully")

# # Use PCA to reduce dimensions for visualization
# pca = PCA(n_components=5)
# pca_result = pca.fit_transform(df_predictor.iloc[:, :-1])

# # Create a new DataFrame for PCA result
# pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2', 'PCA3', "PCA4", "PCA5"])
# pca_df['Cluster'] = clusters

# print(pca_df.head(10))
# pca_df.to_csv('data\\pca_result.csv', index=False)

# # Plot the clusters
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100)
# plt.title('K-means Clustering of Iris Dataset')
# plt.show()