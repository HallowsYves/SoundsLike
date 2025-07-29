import streamlit as st
import numpy as np
import json
from sounds_like_utils import find_similar_songs, find_song_with_fuzzy_matching
from ner.model.pipeline_ner import ner_pipeline
from data_utils import load_data
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


# Load models and data once
@st.cache_resource
def load_model_and_data():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    df_scaled_features = load_data("data/scaled_data.csv", index=True)
    df_song_info = load_data("data/song_data.csv", index=True)
    song_embed = np.load("data/song_embeddings.npy")
    song_embeddings = normalize(song_embed)
    scaled_emotion_means = np.load("data/emotion_vectors.npy")
    with open("data/emotion_labels.json", "r") as f:
        emotion_labels = json.load(f)
    return embedder, df_scaled_features, df_song_info, song_embeddings, scaled_emotion_means, emotion_labels

# App Setup
st.set_page_config(page_title="SoundsLike", layout="wide")
st.title("🎵 SoundsLike: Music Recommendation Engine")
st.caption("Generate music recommendations from natural language prompts like *'sad songs like Moon by Kanye West'*")

# Load everything
embedder, df_scaled_features, df_song_info, song_embeddings, scaled_emotion_means, emotion_labels = load_model_and_data()

# Prompt Input
with st.container():
    st.subheader("💬 Enter Your Prompt")
    user_prompt = st.text_input("What vibe are you going for?", placeholder="e.g. sad songs like Moon by Kanye West")
    num_recs = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
    print(f"Test 1: User Prompt: {user_prompt}")

    if st.button("🔍 Find Songs") and user_prompt.strip():
        
        # Attempt Fuzzy Matching
        print(f"Test 2: User Prompt: {user_prompt}")
        exact_match = find_song_with_fuzzy_matching(user_prompt, df_song_info, ner_pipeline)
        prompt_for_engine = user_prompt
        print(f"Test 3: User Prompt: {user_prompt}")

        if exact_match is not None:
            matched_title = exact_match['Song']
            st.success(f"Found a direct match: {matched_title}. finding similar songs...")
            prompt_for_engine = matched_title
        else:
            st.info("No exact title found. searching by vibe...")

        print(f"Test 4: User Prompt: {user_prompt}")
        print(f"Test 5: User Prompt/Prompt for engine: {prompt_for_engine}")
        result = find_similar_songs(
            user_prompt=user_prompt,
            input_song=exact_match,
            num_recommendations=num_recs,
            ner_pipeline=ner_pipeline,
            embedder=embedder,
            df_scaled_features=df_scaled_features,
            df_song_info=df_song_info,
            song_embeddings=song_embeddings,
            scaled_emotion_means=scaled_emotion_means,
            emotion_labels=emotion_labels
        )

        if result:
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
            #with col1:
                #st.image(main_song["album_art"], width=140)
            with col2:
                st.markdown(f"**{main_song['title']}** by **{main_song['artist']}**")
                st.markdown(f"**Score**: {main_song['score']:.2f}")
                st.image(main_song["radar_chart"], caption="Your Input vs Song Features", use_container_width=True)

            # Recommended Songs
            st.markdown("---")
            st.markdown("### 🎶 Recommended Songs")
            for rec in recs:
                col1, col2 = st.columns([1, 3])
                #with col1:
                    #st.image(rec["album_art"], width=100)
                with col2:
                    st.markdown(f"**{rec['title']}** by **{rec['artist']}**")
                    st.markdown(f"**Score**: {rec['score']:.2f}")
                    st.image(rec["radar_chart"], use_container_width=True)
        else:
            st.error("Sorry, could not find any recommendations. Please try another prompt.")
