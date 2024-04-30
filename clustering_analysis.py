import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

# Function to load and return the cleaned data
def load_cleaned_data(filepath):
    return pd.read_csv(filepath)

# Function to detect and handle outliers
def detect_outliers(df, columns):
    for column in columns:
        z_scores = np.abs(stats.zscore(df[column]))
        df = df[(z_scores < 3)]
    return df

# Function to perform clustering analysis
def clustering_analysis(df, features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(df[features])
    df['cluster'] = kmeans.labels_
    return df, kmeans

# Load the cleaned data
df = load_cleaned_data('C:/Users/Jamie/datasets/stellar-classification/Star99999_cleaned.csv')

# Detect outliers in the B-V and Vmag columns
df = detect_outliers(df, ['B-V', 'Vmag'])

# Perform clustering analysis
features_to_cluster = ['B-V', 'Vmag']
df, kmeans_model = clustering_analysis(df, features_to_cluster, n_clusters=3)

# Visualizations
output_dir = "C:/Users/Jamie/datasets/stellar-classification/viz/"

# Visualize and save potential outliers for B-V Color Index
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['B-V'])
plt.title('Boxplot for B-V Color Index')
plt.savefig(output_dir + 'B-V_Boxplot_Outliers.png')
plt.close()

# Visualize and save clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['B-V'], df['Vmag'], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.title('Clustering on B-V Color Index and Vmag')
plt.xlabel('B-V Color Index')
plt.ylabel('Vmag')
plt.savefig(output_dir + 'Clustering_BV_Vmag.png')
plt.close()

# Save the processed data with cluster labels
processed_filepath = 'C:/Users/Jamie/datasets/stellar-classification/Star99999_processed.csv'
df.to_csv(processed_filepath, index=False)
print(f"Processed data saved to {processed_filepath}")

# Output basic cluster centroids
print("Cluster centroids:")
print(kmeans_model.cluster_centers_)
