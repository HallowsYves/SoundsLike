import pandas as pd
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