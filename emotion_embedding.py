import numpy as np
import pandas as pd
from data_utils import load_data
from sklearn.preprocessing import StandardScaler

"""
    Gets all the features that were used to identify songs, gets their emotion column, and
    the mean of the songs in there. Removes any rows that aren't needed. Then embeds them
    and puts it into a numpy array, creating a txt file for the indexes
"""
df = load_data("data/clean_data.csv", index=True)

features = ["Positiveness", "Danceability", "Energy", "Popularity", "Liveness", "Acousticness", "Instrumentalness"]
emotion_vectors = df.groupby("Emotion")[features].mean()

exclude_emotions = {"thirst", "pink", "interest", "confusion", "angry", "True", "Love"}
filtered_emotions = emotion_vectors[~emotion_vectors.index.isin(exclude_emotions)]

emotion_vector_array = filtered_emotions.values
np.save("data/emotion_means.npy", emotion_vector_array)

with open("data/emotion_labels.txt", "w") as f:
    for label in filtered_emotions.index:
        f.write(label.lower().strip() + "\n")

#Scale it now
emotion_means = np.load("data/emotion_means.npy")
scaler = StandardScaler()
scaler.fit(df[features])
scaled_emotion_means = scaler.transform(emotion_means)
np.save("data/scaled_emotion_means.npy", scaled_emotion_means)
