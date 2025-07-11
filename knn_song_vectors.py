import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
from data_utils import load_data

df_features = load_data('data/scaled_data.csv', index=True)
df_song_info = load_data('data/song_data.csv', index=True)

assert df_song_info.index.equals(df_features.index), "Index mismatch!"


knn = NearestNeighbors(n_neighbors=5)
knn.fit(df_features)

song_name = "Put You On"
matching = df_song_info[df_song_info["Song"] == song_name]

if matching.empty:
    print("Song not found")
else:
    song_index = matching.index[0]
    song_vector = [df_features.loc[song_index].values]

    print(f" Song: {song_name} | Vector: {song_vector}")

    distances, indices = knn.kneighbors(song_vector)

    recommendation_info = df_song_info.iloc[indices[0]].copy()
    recommendation_features = df_features.iloc[indices[0]].copy()
    results = pd.concat([recommendation_info, recommendation_features], axis=1)

    print(f" Original Song: {song_name} | Vector: {song_vector}")
    print(results)

# Radar Chart
    def create_radar_chart(vectors, labels, features):
        num_vars = len(features)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        for vec, label in zip(vectors, labels):
            values = vec.tolist()
            values += values[:1]  # Complete loop
            ax.plot(angles, values, label=label)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features)
        ax.set_title(f"Feature Comparison: {song_name} & Neighbors")
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.show()

    # Get vectors and labels for radar
    features = ['Positiveness_T', 'Danceability_T', 'Energy_T', 'Popularity_T', 'Liveness_T', 'Acousticness_T', 'Instrumentalness_T']
    vectors = recommendation_features[features].values
    labels = recommendation_info['Song'].values

    create_radar_chart(vectors, labels, features)
