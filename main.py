import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from ner.model.pipeline_ner import ner_pipeline
from data_utils import load_data

# Load models and data 
embedder = SentenceTransformer("all-MiniLM-L6-v2")
df_scaled_features = load_data("data/scaled_data.csv", index=True)
df_song_info = load_data("data/song_data.csv", index=True)
song_embed = np.load("data/song_embeddings.npy")
song_embeddings = normalize(song_embed)
scaled_emotion_means = np.load("data/scaled_emotion_means.npy")
with open("data/emotion_labels.txt", "r") as f:
    emotion_labels = [line.strip() for line in f.readlines()]

assert df_song_info.index.equals(df_scaled_features.index), "Index mismatch!"
assert len(df_song_info) == len(song_embeddings), "Mismatch in song info and embeddings"

# Helper functions 
def clean_bert_output(text: str) -> str:
    """Cleans bert text

    Removes anything between [] because of [CLS] in tokens.
    Adds words together the are taken apart using 
    WordPiece tokenization, which start with ##

    Arg:
        text: tokenized text from NER pipeline
    Returns:
        Clean text with no past token marks
    """
    if not text:
        return ""
    text = re.sub(r"\[.*?\]", "", text)
    tokens = text.strip().split()
    cleaned = []
    for token in tokens:
        if token.startswith("##") and cleaned:
            cleaned[-1] += token[2:]
        else:
            cleaned.append(token)
    return " ".join(cleaned)

def get_song_vector(song_name, artist_name):
    """Finds the inputs closest song vector

    Creates an embedded query based on song and/or artist.
    Find's the closest match using cosine and grabs the index.
    The index is used on the song database to get its vector.

    Args:
        song_name: a string that is a song
        artist_name: a string that is an artist
    Returns:
        A vector made up of the chosen features
    """
    if song_name or artist_name:
        query = f"{song_name} by {artist_name}" if song_name and artist_name else song_name or artist_name
        embedding = embedder.encode(query, normalize_embeddings=True)
        sims = cosine_similarity([embedding], song_embeddings)[0]
        idx = np.argmax(sims)
        matched_index = df_song_info.index[idx]
        vector = df_scaled_features.loc[matched_index].values
        info = df_song_info.iloc[idx]
        print(f"\nBest match: {info['Song']} by {info['Artist(s)']} (cos sim: {sims[idx]:.3f})")
        print(f"Matched index: {matched_index}")
        print(f"Scaled vector length: {len(vector)}")
        print(f"Vector sample: {vector[:5]}")
        return vector
    print("No song or artist provided, using fallback vector.")
    return np.zeros(df_scaled_features.shape[1])

def get_emotion_vector(mood):
    """Finds the inputs closest emotion vector
    
    Embeds the mood and labels, normalizing their vectors.
    Find's the closest match using cosine and grabs the index.
    The index is used on the emotion database to get its vector.

    Args:
        mood: a string that is an emotion
    Returns:
        A vector made up of the chosen features
    """
    if mood:
        mood_embedding = embedder.encode(mood, normalize_embeddings=True)
        label_embeddings = [embedder.encode(label, normalize_embeddings=True) for label in emotion_labels]
        sims = cosine_similarity([mood_embedding], label_embeddings)[0]
        idx = np.argmax(sims)
        print(f"Mapped '{mood}' to closest emotion: {emotion_labels[idx]} (cos sim: {sims[idx]:.3f})")
        return scaled_emotion_means[idx]
    print("No mood provided, using neutral vector.")
    return np.zeros(scaled_emotion_means.shape[1])

def run_knn(query_vector, k=5):
    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(df_scaled_features)
    distances, indices = knn.kneighbors([query_vector])
    return indices

def plot_pca(query_vector, indices):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled_features)
    test_2D = pca.transform([query_vector])
    neighbors_2D = pca_result[indices[0]]

    plt.figure(figsize=(10, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.2, label='All Songs', color='gray')
    plt.scatter(neighbors_2D[:, 0], neighbors_2D[:, 1], alpha=0.2, s=100, label='Nearest Neighbors', color='green')
    plt.scatter(test_2D[:, 0], test_2D[:, 1], alpha=0.2, label='Your Prompt', color='red')
    plt.title("KNN Visualization (PCA-Reduced to 2D)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_radar_chart(vectors, labels, features, song_name):
    num_vars = len(features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for vec, label in zip(vectors, labels):
        values = vec.tolist() if not isinstance(vec, list) else vec
        values += values[:1]
        ax.plot(angles, values, label=label)
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_title(f"Feature Comparison: {song_name} & Neighbors")
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

def find_similar_songs(user_prompt, num_recommendations=5):
    entities = ner_pipeline(user_prompt)
    song = clean_bert_output(entities.get("song"))
    artist = clean_bert_output(entities.get("artist"))
    mood = entities.get("mood")
    print(f"song: {song}\nartist: {artist}\nmood: {mood}")

    song_vec = get_song_vector(song, artist)
    emotion_vec = get_emotion_vector(mood)

    assert song_vec.shape == emotion_vec.shape, "Mismatch in feature vector dimensions"

    combined_vec = song_vec + emotion_vec
    print(f"\nQuery vector shape: {combined_vec.shape}")
    print(f"Combined vector sample: {combined_vec}")

    indices = run_knn(combined_vec, num_recommendations)
    plot_pca(combined_vec, indices)

    print("\nRecommended songs:")
    for idx in indices[0][1:]:
        row = df_song_info.iloc[idx]
        print(f"- {row['Song']} by {row['Artist(s)']}")

    features = ['Positiveness_T', 'Danceability_T', 'Energy_T', 'Popularity_T', 'Liveness_T', 'Acousticness_T', 'Instrumentalness_T']
    trimmed_vec = combined_vec[:len(features)]
    vectors = [trimmed_vec] + df_scaled_features.iloc[indices[0][1:]][features].values.tolist()
    labels = ['Your Input'] + df_song_info.iloc[indices[0][1:]]['Song'].tolist()
    create_radar_chart(vectors, labels, features, song)

if __name__ == "__main__":
    prompt = input("Enter a song prompt (e.g., 'give me songs like Moon by Kanye West'): ")
    find_similar_songs(prompt)
