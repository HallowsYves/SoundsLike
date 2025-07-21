import os
from ner.model.pipeline_ner import ner_pipeline
from data_utils import load_data
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import re

# Load emotion means and labels
emotion_means = np.load("data/emotion_means.npy")
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
    song_vector = None


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
        # Embed the raw mood string using your model
        mood_embedding = embedder.encode(mood, normalize_embeddings=True)

        # Compare the mood embedding to each emotion vector (which you already have as emotion_means)
        embedded_labels = [embedder.encode(label) for label in emotion_labels]  # All 384-dim

        # Step 2: Compare NER-extracted mood with emotion label embeddings
        sims = cosine_similarity([mood_embedding], embedded_labels)[0]  # Now shapes match

        # Step 3: Find best match
        best_match_idx = np.argmax(sims)
        matched_label = emotion_labels[best_match_idx]
        emotion_vector = emotion_means[best_match_idx]

        print(f"Mapped '{mood}' to closest emotion: {matched_label} (cos sim: {sims[best_match_idx]:.3f})")
    else:
        print("No mood provided, using neutral vector.")
        emotion_vector = np.zeros(emotion_means.shape[1])

    combined_vector = np.concatenate([song_vector, emotion_vector])
    print(f"\nQuery vector shape: {combined_vector.shape}")

    # Combine all vectors in dataset
    all_combined = np.concatenate([
        df_scaled_features.values,
        np.tile(emotion_vector, (df_scaled_features.shape[0], 1))
    ], axis=1)

    # Run KNN
    knn = NearestNeighbors(n_neighbors=num_recommendations + 1)
    knn.fit(all_combined)
    distances, indices = knn.kneighbors([combined_vector])
    
    print("\nRecommended songs:")
    for idx in indices[0][1:]:
        row = df_song_info.iloc[idx]
        print(f"- {row['Song']} by {row['Artist(s)']}")


if __name__ == "__main__":
    prompt = input("Enter a song prompt (e.g., 'give me songs like Moon by Kanye West'): ")
    find_similar_songs(prompt)
