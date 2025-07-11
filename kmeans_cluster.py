import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from data_utils import load_data

df_scaled = load_data('data/scaled_data.csv', index=True)

kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

df_scaled['Cluster'] = clusters

pca = PCA(n_components=2)
reduced = pca.fit_transform(df_scaled.drop('Cluster', axis=1))

plt.scatter(reduced[:, 0], reduced[:, 1], c=df_scaled['Cluster'], cmap='tab10')
plt.title("KMeans Clustering (k=6)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()