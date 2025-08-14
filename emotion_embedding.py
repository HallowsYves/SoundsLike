import numpy as np
import pandas as pd
from data_utils import load_data
from sklearn.preprocessing import StandardScaler

"""Uses the mean of Emotion features to be scaled for a numpy array

    Groups each individual emotion by its features and grabs their means.
    Filters out any emotions not needed, and creates a base numpy array
    along with a key txt file. Then using the base .npy, it scales it 
    according to how the songs and artists were scaled and saves the new
    vectors into a numpy array.

"""
df = load_data("data/clean_data.csv", index=True)

features = ["Positiveness", "Danceability", "Energy", "Popularity", "Liveness", "Acousticness", "Instrumentalness"]
emotion_vectors = df.groupby("Emotion")[features].mean()

# Could maybe be added into the clean section?
exclude_emotions = {"thirst", "pink", "interest", "confusion", "angry", "True", "Love"}
filtered_emotions = emotion_vectors[~emotion_vectors.index.isin(exclude_emotions)]

emotion_vector_array = filtered_emotions.values
np.save("data/emotion_means.npy", emotion_vector_array)

with open("data/emotion_labels.txt", "w") as f:
    for label in filtered_emotions.index:
        f.write(label.lower().strip() + "\n")

#Scale it now
# One thing i would look into is the scaling measure to the original
emotion_means = np.load("data/emotion_means.npy")
scaler = StandardScaler()
scaler.fit(df[features])
scaled_emotion_means = scaler.transform(emotion_means)
np.save("data/scaled_emotion_means.npy", scaled_emotion_means)
