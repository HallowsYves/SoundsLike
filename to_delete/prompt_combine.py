import numpy as np
import pandas as pd
from ner.model.pipeline_ner import ner_pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

"""
    Grabs the entity info, embeds song and artist to compare it to the embeddings in the dataset.
    Uses the index to get the features vector set. For mood it just compares for a similar str and
    uses the index to get the emotion vector. Then it just combines. 
    Just for notice, the emotion vectors are not scaled, this is just to see them combine
"""
# Load everything once
song_embeddings = np.load("data/song_embeddings.npy")
scaled_song_features = pd.read_csv("data/scaled_data.csv").values
emotion_vectors = np.load("data/scaled_emotion_means.npy")
emotion_labels = [e.strip() for e in open("data/emotion_labels.txt").readlines()]
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

def prompt_to_combined_vector(prompt, ner_pipeline):
    # Step 1: Run NER
    entities = ner_pipeline(prompt)
    mood = entities.get("mood")
    song = entities.get("song")
    artist = entities.get("artist")

    # Step 2: Embed the song+artist and find closest match
    song_query = f"{song} by {artist}"
    song_embedding = sentence_model.encode(song_query)

    similarities = cosine_similarity([song_embedding], song_embeddings)[0]
    best_index = np.argmax(similarities)

    # Step 3: Get corresponding scaled song features
    song_vector = scaled_song_features[best_index]

    # Step 4: Match mood to the closest known emotion (optional: embed)
    if mood and mood.lower() in emotion_labels:
        mood_index = emotion_labels.index(mood.lower())
        mood_vector = emotion_vectors[mood_index]
    else:
        mood_vector = np.zeros(emotion_vectors.shape[1])  # or omit

    # Step 5: Combine both vectors
    combined_vector = np.concatenate([song_vector, mood_vector])

    return combined_vector, best_index

query = "I want something upbeat like Blinding Lights by The Weeknd"
vector, index = prompt_to_combined_vector(query, ner_pipeline)
print("Combined query vector shape:", vector.shape)