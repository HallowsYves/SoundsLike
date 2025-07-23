import os
from ner.model.pipeline_ner import ner_pipeline
from data_utils import load_data
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import re

"""
    Cleans up the entries using clean_bert. Checks if there is anything for them(mood, song, or artist).
    Finds the index from embedded searches, which is used to get the scaled vector. Then combines them,
    running the input through KNN to find the songs.

    *One thing to note which is strange, when i ran "moon by kanye" its giving me songs that include "earth" or "world"
    that doesn't feel like it has much to do with the vector features, so something to look at. Maybe its using the actual
    words as the vector instead of the features.
"""
# Load emotion means and labels
scaled_emotion_means = np.load("data/scaled_emotion_means.npy")
with open("data/emotion_labels.txt", "r") as f:
    emotion_labels = [line.strip() for line in f.readlines()]

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load data
df_scaled_features = load_data("data/scaled_data.csv", index=True)
df_song_info = load_data("data/song_data.csv", index=True)
song_embed = np.load("data/song_embeddings.npy")
song_embeddings = normalize(song_embed)

assert df_song_info.index.equals(df_scaled_features.index), "Index mismatch!"
assert len(df_song_info) == len(song_embeddings), "Mismatch between song info and embeddings"

def clean_bert_output(text: str) -> str:
    if not text:
        return ""
    # Remove special tokens like [SEP], [CLS], etc.
    text = re.sub(r"\[.*?\]", "", text)

    # Merge subwords like '##rell' → join to previous word
    tokens = text.strip().split()
    cleaned = []
    for token in tokens:
        if token.startswith("##") and cleaned:
            cleaned[-1] += token[2:]
        else:
            cleaned.append(token)
    return " ".join(cleaned)

def find_similar_songs(user_prompt, num_recommendations=5):
    entities = ner_pipeline(user_prompt)
    song_name = entities.get("song")
    artist_name = entities.get("artist")
    mood = entities.get("mood")

    song_name = clean_bert_output(song_name)
    artist_name = clean_bert_output(artist_name)
    
    print(f"song: {song_name}")
    print(f"artist: {artist_name}")
    print(f"mood: {mood}")


    if song_name or artist_name:
        if song_name and artist_name:
            query_text = f"{song_name} by {artist_name}"
        elif song_name:
            query_text = song_name
        else:
            query_text = artist_name
        
        query_embedding = embedder.encode(query_text, normalize_embeddings=True)
        sims = cosine_similarity([query_embedding], song_embeddings)[0]
        best_idx = np.argmax(sims)
        best_row = df_song_info.iloc[best_idx]

        matched_index = df_song_info.index[best_idx]
        song_vector = df_scaled_features.loc[matched_index].values

        print(f"\nBest match: {best_row['Song']} by {best_row['Artist(s)']} (cos sim: {sims[best_idx]:.3f})")
        print(f"Matched index: {matched_index}")
        print(f"Scaled vector length: {len(song_vector)}")
        print(f"Vector sample: {song_vector[:5]}")

    else:
        print("No song or artist provided, using fallback vector.")
        song_vector = np.zeros(df_scaled_features.shape[1])

    # Handle emotion embedding
    if mood:
        mood_embedding = embedder.encode(mood, normalize_embeddings=True)

        embedded_labels = [embedder.encode(label) for label in emotion_labels] 

        sims = cosine_similarity([mood_embedding], embedded_labels)[0]

        best_match_idx = np.argmax(sims)
        matched_label = emotion_labels[best_match_idx]
        emotion_vector = scaled_emotion_means[best_match_idx]

        print(f"Mapped '{mood}' to closest emotion: {matched_label} (cos sim: {sims[best_match_idx]:.3f})")
        print(f"Best match: {matched_label}")
        print(f"Vector smaple: {emotion_vector}")
    else:
        print("No mood provided, using neutral vector.")
        emotion_vector = np.zeros(scaled_emotion_means.shape[1])

    assert song_vector.shape == emotion_vector.shape, "Mismatch in feature vector dimensions"
    
    combined_vector = song_vector + emotion_vector
    print(f"\nQuery vector shape: {combined_vector.shape}")
    print(f"Combined vector smaple: {combined_vector}")

    # Run KNN
    knn = NearestNeighbors(n_neighbors=num_recommendations + 1)
    knn.fit(df_scaled_features)
    distances, indices = knn.kneighbors([combined_vector])
    
    # Plot KNN

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled_features)

    # Get 2D positions for neighbors and example
    test_2D = pca.transform([combined_vector])
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

    print("\nRecommended songs:")
    for idx in indices[0][1:]:
        row = df_song_info.iloc[idx]
        print(f"- {row['Song']} by {row['Artist(s)']}")


    def create_radar_chart(vectors, labels, features):
            num_vars = len(features)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

            for vec, label in zip(vectors, labels):
                values = vec if isinstance(vec, list) else vec.tolist()
                values += values[:1]  # Complete loop
                ax.plot(angles, values, label=label)
                ax.fill(angles, values, alpha=0.1)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(features)
            ax.set_title(f"Feature Comparison: {song_name} & Neighbors")
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.tight_layout()
            plt.show()

    recommended_indices = indices[0][1:]

    features = ['Positiveness_T', 'Danceability_T', 'Energy_T', 'Popularity_T', 'Liveness_T', 'Acousticness_T', 'Instrumentalness_T']
    input_vector_trimmed = combined_vector[:len(features)]
    vectors = [input_vector_trimmed] + df_scaled_features.iloc[recommended_indices][features].values.tolist()
    labels = ['Your Input'] + df_song_info.iloc[recommended_indices]['Song'].tolist()

    
    create_radar_chart(vectors, labels, features)


if __name__ == "__main__":
    prompt = input("Enter a song prompt (e.g., 'give me songs like Moon by Kanye West'): ")
    find_similar_songs(prompt)
