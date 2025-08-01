import streamlit as st
import json
from sounds_like_utils import find_similar_songs
from ner.model.pipeline_ner import ner_pipeline
from data_utils import load_data
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import normalize
from load_from_s3 import load_csv_from_s3, load_json_from_s3, load_binary_from_s3

# Load models and data once
@st.cache_resource
def load_model_and_data():
    # Load embedder locally (model download from Hugging Face)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Load CSVs from S3
    df_scaled_features = load_csv_from_s3("scaled_data.csv", index_col=0)
    df_song_info = load_csv_from_s3("song_data.csv", index_col=0)

    # Load numpy arrays from S3 (binary)
    song_embed_bytes = load_binary_from_s3("song_embeddings.npy")
    song_embed = np.load(song_embed_bytes)
    song_embeddings = normalize(song_embed)

    scaled_emotion_bytes = load_binary_from_s3("emotion_vectors.npy")
    scaled_emotions = np.load(scaled_emotion_bytes)

    # Load JSON from S3
    emotion_labels = load_json_from_s3("emotion_labels.json")

    return embedder, df_scaled_features, df_song_info, song_embeddings, scaled_emotions, emotion_labels

# App Setup
st.set_page_config(page_title="Playlist Prompter", layout="wide")
st.title("🎵 Playlist Prompter")
st.caption("Generate music recommendations from natural language prompts like *'sad songs like Moon by Kanye West'*")

# Load everything
embedder, df_scaled_features, df_song_info, song_embeddings, scaled_emotions, emotion_labels = load_model_and_data()

# Prompt Input
with st.container():
    st.subheader("💬 Enter Your Prompt")
    user_prompt = st.text_input("What vibe are you going for?", placeholder="e.g. sad songs like Moon by Kanye West")
    num_recs = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

    if st.button("🔍 Find Songs") and user_prompt.strip():
        result = find_similar_songs(
            user_prompt=user_prompt,
            num_recommendations=num_recs,
            ner_pipeline=ner_pipeline,
            embedder=embedder,
            df_scaled_features=df_scaled_features,
            df_song_info=df_song_info,
            song_embeddings=song_embeddings,
            scaled_emotion_means=scaled_emotions,
            emotion_labels=emotion_labels
        )

        main_song = result["main_song"]
        recs = result["similar_songs"]

        # Detected Entities
        st.markdown("### 🧠 Detected Entities")
        st.markdown(f"- {result['song_match_info']}")
        st.markdown(f"- {result['artist_match_info']}")
        st.markdown(f"- {result['mood_match_info']}")

        # Main Song
        st.markdown("---")
        st.markdown("### 🎯 Closest Match")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(main_song["album_art"], width=140)
        with col2:
            st.markdown(f"**{main_song['title']}** by **{main_song['artist']}**")
            st.markdown(f"**Score**: {main_song['score']:.2f}")
            st.image(main_song["radar_chart"], caption="Your Input vs Song Features", use_container_width=True)

        # Recommended Songs
        st.markdown("---")
        st.markdown("### 🎶 Recommended Songs")
        for rec in recs:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(rec["album_art"], width=100)
            with col2:
                st.markdown(f"**{rec['title']}** by **{rec['artist']}**")
                st.markdown(f"**Score**: {rec['score']:.2f}")
                st.image(rec["radar_chart"], use_container_width=True)
