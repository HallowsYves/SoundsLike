import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import load_and_clean_data
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load Data
df_numeric, song_info = load_and_clean_data()

if df_numeric is not None:
    # Scale Data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_numeric)

    # Run K-means
    chosen_k = 6
    k_means = KMeans(n_clusters=chosen_k, init='k-means++', n_init=10, random_state=42)
    song_info['Mood Cluster'] = k_means.fit_predict(scaled_features)

    # Analyze Clusters
    df_numeric['Mood Cluster'] = song_info['Mood Cluster']
    cluster_profiles = df_numeric.groupby('Mood Cluster').mean()
    print("\n--- Average Audio Features for Each Mood Cluster ---")
    print(cluster_profiles)

# Graph Clusters

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
pca_df['Mood Cluster'] = song_info['Mood Cluster']

plt.figure(figsize=(12,8))
sns.scatterplot(
    x="PC1", y="PC2",
    hue="Mood Cluster",
    palette=sns.color_palette("hsv", n_colors=6),
    data=pca_df,
    legend="full",
    alpha=0.7
)

plt.title("2D Clusters using PCA")
plt.xlabel('Principal Component 1 (Captures most variance)')
plt.ylabel('Principal Component 2 (Captures second-most variance)')
plt.show()