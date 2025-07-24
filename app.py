import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize
from data_utils import load_data
from sentence_transformers import SentenceTransformer



@st._cache_resource
def load_model_and_data():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    df_scaled_features = load_data("data/scaled_data.csv", index=True)
    df_song_info = load_data("data/song_data.csv", index=True)
    song_embed = np.load("data/song_embeddings.npy")
    song_embeddings = normalize(song_embed)
    scaled_emotion_means = np.load("data/scaled_emotion_means.npy")

    with open("data/emotion_labels.txt", "r") as file:
        emotion_lables = [line.strip() for line in file.readlines()]
    return embedder, df_scaled_features, df_song_info, song_embeddings, scaled_emotion_means, emotion_lables

embedder, df_scaled_features, df_song_info, song_embeddings, scaled_emotion_means, emotion_labels = load_model_and_data()