from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from data_utils import load_data
from tqdm import tqdm

df = load_data('data/song_data.csv', index=True)
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_song_artist(row):
    song_artist_str = f"{row['Song']} by {row['Artist(s)']}"
    return model.encode(song_artist_str)

tqdm.pandas()

df["embedding"] = df.progress_apply(embed_song_artist, axis=1)

embeddings_array = np.vstack(df["embedding"].values)
np.save("data/song_embeddings.npy", embeddings_array)

embeddings = np.load("data/song_embeddings.npy")

assert len(df) == len(embeddings), "mismatch between rows and embeddings"