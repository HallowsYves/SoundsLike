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

song_name = "Put You On"
matching = df_song_info[df_song_info["Song"] == song_name].index

if matching.empty:
    print("Song not found")
else:
    song_index = matching[0]
    song_vector = [df_features.loc[song_index].values]

    print(f" Song: {song_name} | Vector: {song_vector}")