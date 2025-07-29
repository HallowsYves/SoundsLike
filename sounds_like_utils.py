import re
import os
import numpy as np
from slugify import slugify
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def clean_bert_output(text: str) -> str:
    """Cleans bert text

    Removes anything between [] because of [CLS] in tokens.
    Adds words together the are taken apart using 
    WordPiece tokenization, which start with ##

    Arg:
        text: tokenized text from NER pipeline
    Returns:
        Clean text with no past token marks
    """
    if not text:
        return ""
    text = re.sub(r"\[.*?\]", "", text)
    tokens = text.strip().split()
    cleaned = []
    for token in tokens:
        if token.startswith("##") and cleaned:
            cleaned[-1] += token[2:]
        else:
            cleaned.append(token)
    return " ".join(cleaned)

def get_song_vector(song_name, artist_name, embedder, df_song_info, song_embeddings, df_scaled_features):
    """Finds the inputs closest song vector

    Creates an embedded query based on song and/or artist.
    Find's the closest match using cosine and grabs the index.
    The index is used on the song database to get its vector.

    Args:
        song_name: a string that is a song
        artist_name: a string that is an artist
    Returns:
        A vector made up of the chosen features
    """
    if song_name or artist_name:
        query = f"{song_name} by {artist_name}" if song_name and artist_name else song_name or artist_name
        embedding = embedder.encode(query, normalize_embeddings=True)
        sims = cosine_similarity([embedding], song_embeddings)[0]
        idx = np.argmax(sims)
        matched_index = df_song_info.index[idx]
        vector = df_scaled_features.loc[matched_index].values
        info = df_song_info.iloc[idx]
        print(f"\nBest match: {info['Song']} by {info['Artist(s)']} (cos sim: {sims[idx]:.3f})")
        print(f"Matched index: {matched_index}")
        print(f"Scaled vector length: {len(vector)}")
        print(f"Vector sample: {vector[:5]}")
        return vector
    print("No song or artist provided, using fallback vector.")
    return np.zeros(df_scaled_features.shape[1])

def get_emotion_vector(mood, embedder, scaled_emotions, emotion_labels):
    """Finds the inputs closest emotion vector
    
    Embeds the mood and labels, normalizing their vectors.
    Find's the closest match using cosine and grabs the index.
    The index is used on the emotion database to get its vector.

    Args:
        mood: a string that is an emotion
    Returns:
        A vector made up of the chosen features
    """
    if mood:
        mood_embedding = embedder.encode(mood, normalize_embeddings=True)
        label_embeddings = [embedder.encode(label, normalize_embeddings=True) for label in emotion_labels]
        sims = cosine_similarity([mood_embedding], label_embeddings)[0]
        idx = np.argmax(sims)
        print(f"Mapped '{mood}' to closest emotion: {emotion_labels[idx]}")
        print(f"Mood Vector: {scaled_emotions[idx]}")
        return scaled_emotions[idx]
    print("No mood provided, using neutral vector.")
    return np.zeros(scaled_emotions.shape[1])

def run_knn(query_vector, df_scaled_features, k=5):
    """Setting up a K Nearest Neighbors graph

    Define how many neighbors you want back, then plot 
    all the points onto the graph. A query is used 
    define the central point and those around.

    Args:
        query_vector: a vector consisting of the features used to fit
        k: the amount of indices to be returned
    Returns:
        The indices of the closest points to the query plot
    """
    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(df_scaled_features)
    distances, indices = knn.kneighbors([query_vector])
    return distances, indices

def plot_pca(query_vector, indices, df_scaled_features):
    """Visualises a 2D plot for the query and indicies

    Uses Principal Component Analysis to turn all the
    vectors into 2D. It has 3 different targets: background (gray),
    query (red), and neighbor (green) points. 

    Args:
        query_vector: a vector consisting of the features used to plot
        indicies: the closest vectors to the query_vector
    Returns:
        Nothing, but a plot does pop out with the points marked 
    """
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled_features)
    test_2D = pca.transform([query_vector])
    neighbors_2D = pca_result[indices[0]]

    fig, ax = plt.subplots(figsize=(10, 6)) # Create figure and axes
    ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.2, label='All Songs', color='gray')
    ax.scatter(neighbors_2D[:, 0], neighbors_2D[:, 1], alpha=0.2, s=100, label='Nearest Neighbors', color='green')
    ax.scatter(test_2D[:, 0], test_2D[:, 1], alpha=0.2, label='Your Prompt', color='red')
    ax.set_title("KNN Visualization (PCA-Reduced to 2D)")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    
    return fig 

def create_radar_chart(vector, title, features, output_dir="output"):
    """Creates a radar chart using the vectors

    Splits a circle beetween the amount of angles. 
    Assigns labels to vectors, which are then graphed
    according to their features.

    Args:
        vectors: the numbers used to chart it
        labels: the song names to each chart
        features: the labels for the xtick
        song_name: used for the title
    Returns:
        Nothing, but a chart shows itself with all the vectors
    """
    num_vars = len(features)

    # Create angle slices
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    if isinstance(vector, np.ndarray):
        values = vector.tolist()
    else:
        values = list(vector)

    values = values + values[:1]  # close the loop
    

    # Init plot
    print(f"[DEBUG] angles length: {len(angles)}")
    print(f"[DEBUG] values length: {len(values)}")

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="blue", linewidth=2)
    ax.fill(angles, values, color="blue", alpha=0.25)
    ax.set_title(title, size=11, pad=20)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_yticklabels([])

    # Save to file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"radar_{slugify(title)}.png"
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)

    return filepath

def find_similar_songs(user_prompt, num_recommendations, ner_pipeline, embedder, df_scaled_features, df_song_info, song_embeddings, scaled_emotion_means, emotion_labels):    
    """Finds similar songs according to the prompt

    Uses the NER pipeline to decipher the entities in the prompt.
    Finds the vectors for each of them, combines them, and runs
    it through KNN to get the most similar songs.

    Args:
        user_prompt: an input that details mood, song, and/or artist
        num_recommendations: the amount of songs the user wants
    Returns:
        Nothing, just terminal prints and matplot plots
    """
    entities = ner_pipeline(user_prompt)
    song = clean_bert_output(entities.get("song"))
    artist = clean_bert_output(entities.get("artist"))
    mood = entities.get("mood")

    song_match_info = f"Detected Song: **{song if song else 'N/A'}**"
    artist_match_info = f"Detected Artist: **{artist if artist else 'N/A'}**"
    mood_match_info = f"Detected Mood: **{mood if mood else 'N/A'}**"

    print(song_match_info)
    print(artist_match_info)
    print(mood_match_info)

    song_vec = get_song_vector(song, artist, embedder, df_song_info, song_embeddings, df_scaled_features)
    emotion_vec = get_emotion_vector(mood, embedder, scaled_emotion_means, emotion_labels)
    assert song_vec.shape == emotion_vec.shape, "Mismatch in feature vector dimensions"

    if np.all(song_vec == 0):
        combined_vec = emotion_vec
    elif np.all(emotion_vec == 0):
        combined_vec = song_vec
    else:
        combined_vec = (song_vec * .7 ) + (emotion_vec *.3)
        
    print(f"\nQuery vector shape: {combined_vec.shape}")
    print(f"Combined vector sample: {combined_vec}")

    print("Running KNN with vector shape:", combined_vec.shape)
    print("Data shape:", df_scaled_features.shape)  

    distances, indices = run_knn(combined_vec, df_scaled_features, num_recommendations)
    print("Distances:", distances)
    print("Indices:", indices)
    
    top_indices = indices[0]
    top_distances = distances[0]

    features = ['Positiveness_T', 'Danceability_T', 'Energy_T', 'Popularity_T', 'Liveness_T', 'Acousticness_T', 'Instrumentalness_T']
    radar_images = []
    for idx in top_indices:
        vec = df_scaled_features.iloc[idx]
        title = df_song_info.iloc[idx]["Artist(s)"]
        radar_path = create_radar_chart(vec, title, features)
        radar_images.append(radar_path)

    main_idx = top_indices[0]
    main_dist = top_distances[0]
    main_song_data = df_song_info.iloc[main_idx]

    main_song = {
        "title": main_song_data['Song'],
        "artist": main_song_data['Artist(s)'],
        "score": 1 - main_dist,
        "album_art": "img/cover_art.jpg",
        "radar_chart": radar_images[0]
    }

    similar_songs = []
    for i, (idx, dist) in enumerate(zip(top_indices[1:], top_distances[1:])):
        song = df_song_info.iloc[idx]
        similar_songs.append({
            "title": song['Song'],
            "artist": song['Artist(s)'],
            "score": 1 - dist,
            "album_art": "img/cover_art.jpg",
            "radar_chart": radar_images[i + 1] 
        })

    return {
        "main_song": main_song,
        "similar_songs": similar_songs,
        "song_match_info":song_match_info,
        "artist_match_info": artist_match_info,
        "mood_match_info": mood_match_info
    }