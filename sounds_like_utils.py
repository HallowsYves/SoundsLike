import re
import os
import numpy as np
from slugify import slugify
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz,process


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

def get_emotion_vector(mood, embedder, scaled_emotion_means, emotion_labels):
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
        print(f"Mood Vector: {scaled_emotion_means[idx]}")
        return scaled_emotion_means[idx]
    print("No mood provided, using neutral vector.")
    return np.zeros(scaled_emotion_means.shape[1])

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

def create_radar_chart(vector1, vector2, title, features,labels=["Your Song", "Recommendation"], output_dir="output"):
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
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

    # Plot Vector 1 (User's Song)
    values1 = vector1.tolist() + vector1.tolist()[:1]
    ax.plot(angles, values1, color="red", linewidth=2, label=labels[0])
    ax.fill(angles, values1, color="red", alpha=0.25)

    # Plot Vector 2 (Recommended Song)
    values2 = vector2.tolist() + vector2.tolist()[:1]
    ax.plot(angles, values2, color="blue", linewidth=2, label=labels[1])
    ax.fill(angles, values2, color="blue", alpha=0.25)

    ax.set_title(title, size=11, pad=20)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_yticklabels([])
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Save to file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"radar_{slugify(title)}.png"
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)

    return filepath

def find_song_with_fuzzy_matching(query, song_df, ner_pipeline, threshold=85):
    """
    Finds best song match using fuzzy tring matching.
    Returns the song (as a series) if a match is found, otherwise None
    """
    entities = ner_pipeline(query)
    song_entity = clean_bert_output(entities.get("song"))
    artist_entity = clean_bert_output(entities.get('artist'))

    if song_entity and artist_entity:
        artist_songs = song_df[song_df['Artist(s)'].str.contains(artist_entity, case=False, na=False)]
        if not artist_songs.empty:
            match = process.extractOne(song_entity, artist_songs['Song'], scorer=fuzz.ratio)
            if match and match[1] > 90: 
                return artist_songs[artist_songs['Song'] == match[0]].iloc[0]


    search_query = song_entity if song_entity else query

    best_match = process.extractOne(search_query, song_df['Song'], scorer=fuzz.token_set_ratio)

    if best_match and best_match[1] >= threshold:
        matched_title = best_match
        return song_df['Song'] == matched_title[0]
    
    return None


def find_similar_songs(user_prompt, input_song, num_recommendations, ner_pipeline, embedder, df_scaled_features, df_song_info, song_embeddings, scaled_emotion_means, emotion_labels):    
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
    song_entity = clean_bert_output(entities.get("song"))
    artist_entity = clean_bert_output(entities.get("artist"))
    mood_entity = entities.get("mood")

    song_for_vec = input_song['Song'] if input_song is not None else song_entity
    artist_for_vec = input_song['Artist(s)'] if input_song is not None else artist_entity
    
    song_vec = get_song_vector(song_for_vec, artist_for_vec, embedder, df_song_info, song_embeddings, df_scaled_features)
    emotion_vec = get_emotion_vector(mood_entity, embedder, scaled_emotion_means, emotion_labels)
    
    if np.all(song_vec == 0):
        combined_vec = emotion_vec
        print("combine = emotion")
    elif np.all(emotion_vec == 0):
        combined_vec = song_vec
        print("combine = song")
    else:
        combined_vec = (song_vec * .7 ) + (emotion_vec *.3)
        print("combine = both")
    
    distances, indices = run_knn(combined_vec, df_scaled_features, num_recommendations + 1)
    top_indices = indices[0]
    
    features = ['Positiveness', 'Danceability', 'Energy', 'Popularity', 'Liveness', 'Acousticness', 'Instrumentalness']
    
    if input_song is not None:
        main_song_data = input_song
        input_graph_vector = df_scaled_features.loc[main_song_data.name].values
    else:
        main_song_data = df_song_info.iloc[top_indices[0]]
        input_graph_vector = combined_vec

    # Create the main song dictionary for the UI
    main_song_vector = df_scaled_features.loc[main_song_data.name].values
    main_song_radar_path = create_radar_chart(input_graph_vector, main_song_vector, f"{main_song_data['Song']} Profile", features)

    main_song_display = {
        "title": main_song_data['Song'],
        "artist": main_song_data['Artist(s)'],
        "score": 1.0 if input_song is not None else (1 - distances[0][0]),
        "album_art": "img/cover_art.jpg",
        "radar_chart": main_song_radar_path
    }

    similar_songs = []
    for idx in top_indices:
        if input_song is not None and df_song_info.iloc[idx]['Song'] == input_song['Song']:
            continue
            
        if input_song is None and idx == top_indices[0]:
            continue

        rec_song_data = df_song_info.iloc[idx]
        rec_vector = df_scaled_features.iloc[idx].values
        
        radar_path = create_radar_chart(input_graph_vector, rec_vector, f"Comparison: {rec_song_data['Song']}", features)
        
        similar_songs.append({
            "title": rec_song_data['Song'],
            "artist": rec_song_data['Artist(s)'],
            "score": 1 - distances[0][np.where(top_indices == idx)[0][0]],
            "album_art": "img/cover_art.jpg",
            "radar_chart": radar_path
        })
        
        if len(similar_songs) == num_recommendations:
            break
            
    return {
        "main_song": main_song_display,
        "similar_songs": similar_songs,
        "song_match_info": f"Detected Song: **{song_for_vec if song_for_vec else 'N/A'}**",
        "artist_match_info": f"Detected Artist: **{artist_for_vec if artist_for_vec else 'N/A'}**",
        "mood_match_info": f"Detected Mood: **{mood_entity if mood_entity else 'N/A'}**"
    }