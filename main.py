import os
from ner.model.pipeline_ner import ner_pipeline
from data_utils import load_data
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def find_similar_songs(user_prompt, num_recommendations=5):
    """
    Extract Song info from user prompt using NER
    """
    entities = ner_pipeline(user_prompt)
    song_name = entities.get("song")
    artist_name = entities.get("artist")

    
    # Load Data
    if not song_name:
        print("Sorry, No song could be identified in your request. Please specify a song title.")
        return

    try:
        df_features = load_data('data/scaled_data.csv', index=True)
        df_song_info = load_data('data/song_data.csv', index=True)

        if df_features is None or df_song_info is None:
            print("Error loading data files. Please ensure 'data/scaled_data.csv' and 'data/song_data.csv' exist.")
            return

    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return


    assert df_song_info.index.equals(df_features.index), "Index mismatch!"

    # Find Vector of specified Song
    matching_songs = df_song_info[df_song_info["Song"].str.lower() == song_name.lower()]

    if matching_songs.empty:
        print(f"Song {song_name} not found in database. Please try a different song.")

    # In case of multiple songs with the same name, go based off of first match.
    song_index = matching_songs.index[0]
    song_vector = [df_features.loc[song_index].values]
    
    print(f"\nFinding songs similar to: {song_name} by {df_song_info.loc[song_index, 'Artist(s)']}\n")

    # Perform K-NN Search
    knn = NearestNeighbors(n_neighbors=num_recommendations + 1)
    knn.fit(df_features)

    distances, indices = knn.kneighbors(song_vector)

    # Get and display recommendations
    recommendation_indices = indices[0][1:]
    recommendations = df_song_info.iloc[recommendation_indices].copy()

    print("Here are some similar songs:")
    for index, row in recommendations.iterrows():
        print(f"- {row['Song']} by {row['Artist(s)']}")


if __name__ == "__main__":
    prompt = input("Enter a song prompt (e.g.., 'give me songs similar to Put You On'): ")
    find_similar_songs(prompt)