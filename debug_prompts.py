import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from data_utils import load_data

"""
    Embeds a query string and returns the top K closest songs from the dataset.

    Args:
        query (str): The user prompt (e.g., "Happy by Pharrell Williams")
        top_k (int): How many closest results to show
    """
model = SentenceTransformer("all-MiniLM-L6-v2")
song_embeddings = np.load("data/song_embeddings.npy")
df = load_data("data/song_data.csv", index=True)

def debug_match(query: str, top_k=5):
    query_embedding = model.encode(query, normalize_embeddings=True)
    
    # Compute cosine similarities
    similarities = cosine_similarity([query_embedding], song_embeddings)[0]
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\n🔍 Top {top_k} matches for: \"{query}\"\n")
    for i, idx in enumerate(top_k_indices):
        row = df.iloc[idx]
        score = similarities[idx]
        
        song = row["Song"]
        artist = row["Artist(s)"]
        info = f"{i+1}.{song} by {artist} — Similarity: {score:.4f}"
        
        print(info)
    print("\n")

debug_match("Happy by Pharrell Williams")
