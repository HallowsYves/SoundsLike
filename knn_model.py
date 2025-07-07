import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


try:
    df_features = pd.read_csv('data/scaled_features.csv', index_col=0)
    print('found dataset')
except FileNotFoundError:
    print("could not find dataset")

try:
    df_song_info = pd.read_csv('data/song_info.csv', index_col=0)
    print('found dataset')
except FileNotFoundError:
    print("could not find dataset")

assert df_song_info.index.equals(df_features.index), "Index mismatch!"

print(df_features.shape[1])
print(df_features.describe())

knn = NearestNeighbors(n_neighbors=5)
knn.fit(df_features)

# #Find a prompting method
test = [[1.0, 0.1, 0.1, .7]] # Positiveness, Dance, Energy, Popularity

distances, indices = knn.kneighbors(test)


recommendations = df_song_info.iloc[indices[0]]
print(recommendations)



# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_features)

# Get 2D positions for neighbors and example
test_2D = pca.transform(test)
neighbors = df_features.iloc[indices[0]]
neighbors_2D = pca.transform(neighbors)


# Plot all songs
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.3, label='All Songs', color='gray')

# Plot neighbors
plt.scatter(neighbors_2D[:, 0], neighbors_2D[:, 1], alpha=0.3, s=100, label='Nearest Neighbors', color='green')

# Plot the test point
plt.scatter(test_2D[:, 0], test_2D[:, 1], alpha=0.3, label='Your Prompt', color='red')

# Label
plt.title("KNN Visualization (PCA-Reduced to 2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()