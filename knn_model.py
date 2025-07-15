from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from data_utils import load_data

"""
    Takes in the scaled vectors (test)
    Computes the distances and indicies from closest to farthest
"""

df_features = load_data('data/scaled_data.csv', index=True)
df_song_info = load_data('data/song_data.csv', index=True)

assert df_song_info.index.equals(df_features.index), "Index mismatch!"

knn = NearestNeighbors(n_neighbors=5)
knn.fit(df_features)

test = [[1.0, 0.1, 0.1, .7, 1.0, 0.2, -0.3]] # Positiveness, Dance, Energy, Popularity, Liveness, Acousticness, Instrumentalness

distances, indices = knn.kneighbors(test)


recommendations = df_song_info.iloc[indices[0]]
print(recommendations)



# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_features)

# Get 2D positions for neighbors and example
test_2D = pca.transform(test)
neighbors_2D = pca_result[indices[0]]


# Plot all songs
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.2, label='All Songs', color='gray')

# Plot neighbors
plt.scatter(neighbors_2D[:, 0], neighbors_2D[:, 1], alpha=0.2, s=100, label='Nearest Neighbors', color='green')

# Plot the test point
plt.scatter(test_2D[:, 0], test_2D[:, 1], alpha=0.2, label='Your Prompt', color='red')

# Label
plt.title("KNN Visualization (PCA-Reduced to 2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()