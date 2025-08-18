import streamlit as st
import numpy as np
import json
from io import BytesIO
import base64

# light, safe imports
from sounds_like_utils import find_similar_songs, find_song_with_fuzzy_matching
from ner.model.pipeline_ner import ner_pipeline
from data_utils import load_data
from hf_utils import load_csv_from_hub, load_json_from_hub, load_binary_from_hub


from spotipy_util import init_spotify, get_spotify_track


def get_spotify_safe():
    try:
        if not st.secrets.get("spotify"):
            return None
        return init_spotify()
    except Exception as e:
        st.sidebar.warning(f"Spotify disabled: {e}")
        return None


@st.cache_resource
def load_model_and_data():
    # heavy imports happen here so we can catch errors in UI
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import normalize

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    df_scaled_features = load_csv_from_hub("scaled_data.csv", index_col=0)
    df_song_info      = load_csv_from_hub("song_data.csv",   index_col=0)

    song_embed_bytes = load_binary_from_hub("song_embeddings.npy")
    song_embed = np.load(BytesIO(song_embed_bytes), allow_pickle=False)
    song_embeddings = normalize(song_embed)

    scaled_emotion_bytes = load_binary_from_hub("emotion_vectors.npy")
    scaled_emotions = np.load(BytesIO(scaled_emotion_bytes), allow_pickle=False)

    emotion_labels = load_json_from_hub("emotion_labels.json")

    return (
        embedder,
        df_scaled_features,
        df_song_info,
        song_embeddings,
        scaled_emotions,
        emotion_labels,
    )


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# --- UI / main ---
try:
    st.set_page_config(page_title="SoundsLike", layout="wide")

    spotify = get_spotify_safe()

    st.title("🎵 SoundsLike: Music Recommendation Engine")
    st.caption("Generate music recommendations from natural language prompts like *'sad songs like Moon by Kanye West'*")

    (embedder, df_scaled_features, df_song_info,
     song_embeddings, scaled_emotion_means, emotion_labels) = load_model_and_data()

    with st.container():
        st.subheader("💬 Enter Your Prompt")
        user_prompt = st.text_input("What vibe are you going for?",
                                    placeholder="e.g. sad songs like Moon by Kanye West")
        num_recs = st.slider("Number of recommendations", 3, 10, 5)

        if st.button("🔍 Find Songs") and user_prompt.strip():
            (result_tuple, closest_match) = find_song_with_fuzzy_matching(user_prompt, df_song_info, ner_pipeline)
            exact_match = result_tuple
            prompt_for_engine = user_prompt

            if exact_match is not None and closest_match is True:
                matched_title = exact_match['Song']
                st.success(f"Found a direct match: {matched_title}. finding similar songs...")
                prompt_for_engine = matched_title
            else:
                st.info("No exact title found. searching by vibe...")

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
                recs = [main_song] + result["similar_songs"]

                st.markdown("### 🧠 Detected Entities")
                st.markdown(f"- {result['song_match_info']}")
                st.markdown(f"- {result['artist_match_info']}")
                st.markdown(f"- {result['mood_match_info']}")
                st.markdown("---")
                st.markdown("### 🎶 Recommended Songs")

                for rec in recs:
                    # only try Spotify if configured
                    track = get_spotify_track(spotify, rec['title'], rec['artist']) if spotify else None
                    ...
                    # (unchanged rendering code)

except Exception as e:
    st.error("Startup error – details below.")
    st.exception(e)
