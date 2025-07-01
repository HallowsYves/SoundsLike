import pandas as pd
import matplotlib as plt
from sklearn.cluster import KMeans

# Load Data
try: 
    df = pd.read_csv('spotify_cleaned.csv')
except FileNotFoundError:
    print("Could not find 'spotify_cleaned.csv'.")
features = df[['Genre',  'Emotion']]


# Run KMeans and fit it to the data
k_means = KMeans(n_clusters=5, init='k-means++', random_state=42)

# Add Clusters to Df
cluster_labels = k_means.fit_predict(features)
df['Cluster'] = cluster_labels

plt.scatter(df['Genre'], df['Emotion'], c=df['Cluster'], cmap='viridis')

centroids = k_means.cluster_centers_

plt.scatter(centroids[:, 0], centroids[:, 1], s=75, marker='X', c='red')

plt.show()