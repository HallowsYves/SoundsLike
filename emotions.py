import json
import numpy as np
from data_utils import load_data

mood_to_index = {
    "joy": 333043,
    "excitement": 277273,
    "contentment": 194885,
    "hype": 458966,
    "energetic": 344808,
    "gratitude": 113345,
    "sadness": 9456,
    "loneliness": 136707,  
    "regret": 411457,
    "melancholy": 263322,
    "heartbreak": 252468,
    "anger": 13546,
    "rage": 347726,
    "jealousy": 245563,
    "frustration": 346994,
    "fear": 172996,
    "insecurity": 383247,
    "dark": 444171,
    "dread": 424493,
    "love": 211736,
    "passion": 444053,
    "comfort": 64832,
    "affection": 91564,
    "dreamy": 39595,
    "romantic": 132456,
    "surprise": 411427,
    "curiosity": 194886,
    "wonder": 103376,
    "shock": 225554,
    "awe": 344670,
    "calm": 56829,
    "chill": 192827,
    "swagger": 229980,
    "cool": 410206,
    "neutral": 88822
}

mood_df = load_data("data/scaled_data.csv", index=True)

features = ["Positiveness_T", "Danceability_T", "Energy_T", "Popularity_T", "Liveness_T", "Acousticness_T", "Instrumentalness_T"]

assert all(col in mood_df.columns for col in features)

emotion_labels = []
emotion_vectors = []

for mood, idx in mood_to_index.items():
    if idx in mood_df.index:
        vec = mood_df.loc[idx, features].values.astype(np.float32)
        emotion_labels.append(mood)
        emotion_vectors.append(vec)
    else:
        print(f"Index {idx} for mood '{mood}' not found in dataset")

emotion_vectors = np.array(emotion_vectors)

with open("data/emotion_labels.json", "w") as f:
    json.dump(emotion_labels, f)

np.save("data/emotion_vectors.npy", emotion_vectors)