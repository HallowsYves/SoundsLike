import re
import os
import numpy as np
from slugify import slugify
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process


def clean_bert_output(text: str) -> str:  
    """
    Cleans bert text output.

    Removes anything between [] because of [CLS] in tokens.
    Adds words together the are taken apart using 
    WordPiece tokenization, which start with ##.

    Arg:
        text (str): Tokenized text from NER pipeline.

    Returns:
        str: Clean text with no past token marks and subwords merged.
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
    """
    Finds the inputs closest song vector.

    Creates an embedded query based on song and/or artist.
    Find's the closest match using cosine and grabs the index.
    The index is used on the song database to get its vector.

    Args:
        song_name (str): The name of the song used for embedding to find the closest vector match.
        artist_name (str): The name of the artist used for embedding to find the closest vector match.
        embedder (SentenceTransformer): The model chosen to embed.
        df_song_info (pd.DataFrame): Composed of all the songs and artists in the dataset. Matched indexes with df_scaled_features.
        song_embeddings (np.ndarray): Contains the embedded combinations of all the artist and songs.
        df_scaled_features (pd.DataFrame): Composed of all the features in the dataset. Matched indexes with df_song_info.
    
    Returns:
        np.ndarray: A vector containing feature values to the best matched song, 
                    or a zero vector if no song or artist was provided.
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
    """
    Finds the inputs closest emotion vector.
    
    Embeds the mood and labels, normalizing their vectors.
    Find's the closest match using cosine and grabs the index.
    The index is used on the emotion database to get its vector.

    Args:
        mood (str): The mood identified from the user's prompt.
        embedder (SentenceTransformer): The model chosen to embed.
        scaled_emotion_means (np.ndarray): An array mapping anchored song indexes to emotions.
        emotion_labels (List[str]): All the sub-emotions to compare to.

    Returns:
        np.ndarray: The vector features of the song mapped to the emotion, 
                    or a zero vector if no mood was provided.
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
    """
    Finds the nearest neigbhors of a query vector using KNN.

    Define how many neighbors you want back, then plot 
    all the points onto the graph. A query is used 
    define the central point and those around.

    Args:
        query_vector (np.ndarray): A vector consisting of the features used as the query point.
        df_scaled_features (pd.DataFrame): Composed of all the features in the dataset, used to fit the model.
        k (int, optional): The amount of neighbors to be returned (default = 5).

    Returns:
        Tuple[np.ndarray, np.ndarray]: contains 2 arrays:
            - distances: The distances to each of the neighbors.
            - indices: The indicies to the neigbors in the dataset.
    """
    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(df_scaled_features)
    distances, indices = knn.kneighbors([query_vector])
    return distances, indices

def distance_to_similarity(distance: float) -> float:
    """Convert a distance value into a similarity score.

    The ``NearestNeighbors`` model returns Euclidean distances where a lower
    value indicates higher similarity.  Subtracting these distances from ``1``
    can yield negative scores because distances are not bounded by ``1``.
    This helper transforms a distance ``d`` into ``1 / (1 + d)``, producing a
    value in the ``0``–``1`` range where ``1`` represents an identical match.

    Args:
        distance (float): The distance value returned from KNN.

    Returns:
        float: A similarity score between ``0`` and ``1``.
    """
    return 1 / (1 + distance)

def plot_pca(query_vector, indices, df_scaled_features):
    """
    Visualises a 2D plot for the query and indicies

    Uses Principal Component Analysis to turn all the
    vectors into 2D. It has 3 different targets: background (gray),
    query (red), and neighbor (green) points. 

    Args:
        query_vector (np.ndarray): A vector consisting of the features used to plot.
        indicies (np.ndarray): The closest vectors to the query_vector.
        df_scaled_features (pandas.Dataframe): Composed of all the features in the dataset, used to fit the graph.

    Returns:
        matplotlib.figure.Figure: A figure containing the PCA plot.
    """
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled_features)
    test_2D = pca.transform([query_vector])
    neighbors_2D = pca_result[indices[0]]

    fig, ax = plt.subplots(figsize=(10, 6))
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

def escape_latex_chars(s):
    """
    Escapes characters that could be misinterpreted by Matplotlib's mathtext engine.
    """
    return re.sub(r'([$%#&{}_])', r'\\\1', s)

def create_radar_chart(vector1, vector2, title, features, labels=["Your Song", "Recommendation"], output_dir="output"):
    """
    Creates a radar chart comparison between 2 vectors.

    Splits a circle beetween the amount of angles. 
    Assigns labels to vectors, which are then graphed
    according to their features.

    Args:
        vector1 (np.ndarray): The comparison song's vector features.
        vector2 (np.ndarray): The main song's vector features.
        title (str): The main song's name.
        features (List[str]): The features to label .
        labels (List[str]): The song names to each chart.
        output_dir (str, optional): Where the file should save (default = "output").

    Returns:
        str: The full file path to the saved radar chart image.
    """

    num_vars = len(features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    values1 = vector1.tolist() + vector1.tolist()[:1]
    values2 = vector2.tolist() + vector2.tolist()[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
  
    fig.patch.set_facecolor("#121212") 
    ax.set_facecolor("#121212")
  
    ax.plot(angles, values1, color="#CCCCCC", linewidth=1.5, label=labels[1])
    ax.fill(angles, values1, color="#CCCCCC", alpha=0.25, zorder=1)
    
    ax.plot(angles, values2, color="#1db954", linewidth=2, label=labels[0])
    ax.fill(angles, values2, color="#1db954", alpha=0.4, zorder=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, size=11, weight='medium', color='white')
    ax.set_yticklabels([])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=10, labelcolor='white')

    ax.spines['polar'].set_visible(False)
    ax.grid(color='white', linestyle='dashed', linewidth=0.6, alpha=0.4)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"radar_{slugify(title)}.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    return filepath


def normalize(text):
    return text.lower().strip().replace("’", "'").replace("`", "'")


def extract_song_artist_from_prompt(prompt):
    """
    Extracts phrases like '2031 by Inner Wave' from prompts like
    'sad songs like 2031 by Inner Wave'.

    Args:
        prompt (str): The full prompt from the user
    
    Returns:
        str: 2 groups, one with the song name, the other with artist name,
             or none if no match is empty.
    """
    match = re.search(r"(?:like\s+)(.+?)\s+by\s+(.+)", prompt, re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    print("Returning none on extract")
    return None, None


def find_song_with_fuzzy_matching(query, song_df, ner_pipeline, threshold=85):
    """
    Attempts to match song and artist using regex, NER, and fuzzy matching.
    
    Using regex, it tries to get the artist and song. If it fails, it uses the NER
    model as a fallback. It has some cases against known errors. Finally it uses 
    fuzzy matching of the song/artist against the song dataframe.

    Args:
        query (str): The input string from the user
        song_df (pandas.Dataframe): Composed of all the songs and artists in the dataset.
        ner_pipeline (Callable): The NER pipeline that extracts "song" and "artist" from the query.
        threshold (int, optional): The minimum matching score for fuzzy (default = 85).
    
    Returns:
        pd.Series: A row from the song_df that best matches the query, 
                   or None if no match is found.
    """
    #  Try regex override 
    structured_song, structured_artist = extract_song_artist_from_prompt(query)

    if structured_song and structured_artist:
        print(f"[Regex Extracted] Song: {structured_song} | Artist: {structured_artist}")
        song_entity = structured_song
        artist_entity = structured_artist
    else:
        # Fallback to NER
        entities = ner_pipeline(query)
        song_entity = clean_bert_output(entities.get("song"))
        artist_entity = clean_bert_output(entities.get("artist"))

    song_entity = normalize(song_entity) if song_entity else ""
    artist_entity = normalize(artist_entity) if artist_entity else ""
    song_df['Song'] = song_df['Song'].str.lower().str.strip()
    song_df['Artist(s)'] = song_df['Artist(s)'].str.lower().str.strip()

    # Step 4: Known problem artists override
    KNOWN_ARTISTS = ["tv girl", "inner wave", "the 1975"]
    for known in KNOWN_ARTISTS:
        if known in normalize(query):
            artist_entity = known
            break

    if song_entity and artist_entity and song_entity in artist_entity:
        print(f"[Heuristic] Ignoring song '{song_entity}' embedded in artist '{artist_entity}'")
        song_entity = ""

    if song_entity and artist_entity:
        artist_songs = song_df[song_df['Artist(s)'].str.contains(artist_entity, case=False, na=False)]
        if not artist_songs.empty:
            match = process.extractOne(
                song_entity,
                artist_songs['Song'],
                scorer=fuzz.token_sort_ratio
            )
            if match and match[1] > 90 and normalize(match[0]) == song_entity:
                matched_rows = artist_songs[artist_songs['Song'] == match[0]]
                if not matched_rows.empty:
                    return matched_rows.iloc[0], True 

    search_query = song_entity if song_entity else query
    best_match = process.extractOne(search_query, song_df['Song'], scorer=fuzz.token_set_ratio)

    if best_match and best_match[1] >= threshold:
        matched_rows = song_df[song_df['Song'] == best_match[0]]
        if not matched_rows.empty:
            return matched_rows.iloc[0], False

    return None, None



def find_similar_songs(user_prompt, input_song, num_recommendations, ner_pipeline, embedder, df_scaled_features, df_song_info, song_embeddings, scaled_emotion_means, emotion_labels):    
    """
    Finds similar songs according to the prompt

    Uses the song, artist, and/or mood entities from either fuzzy matching or NER as points.
    It gets the vector points for the entities, combining them and inserting it into KNN.
    Gathers each similar song's information: song/artist, similarity score, and radar_charts.

    Args:
        user_prompt (str): An input that details mood, song, and/or artist.
        input_song (pd.Series): A matched song row if Fuzzy was successful.
        num_recommendations (int): The amount of songs the user wants.
        ner_pipeline (Callable): A named entity recognition model which identifies song, artist, and mood.
        embedder (SentenceTransformer): The model chosen to embed.
        df_scaled_features (pd.DataFrame): Scaled song features used in similarity calculations.
        df_song_info (pd.DataFrame): Raw song metadata including titles and artist names.
        song_embeddings (np.ndarray): Embedded representations of all songs in the dataset.
        scaled_emotion_means (np.ndarray): Pre-computed emotion vectors for reference moods.
        emotion_labels (List[str]): List of emotion label strings used to match moods.
        
    Returns:
        Dict[str, Any]: A dictionary with:
            - 'main_song': dict with title, artist, similarity score, radar chart path
            - 'similar_songs': list of dicts with title, artist, score, radar chart path
            - 'song_match_info': formatted string for display
            - 'artist_match_info': formatted string for display
            - 'mood_match_info': formatted string for display
    """
    entities = ner_pipeline(user_prompt)
    mood_entity = entities.get("mood")
    
    if input_song is not None:
        song_entity = input_song['Song']
        artist_entity = input_song['Artist(s)']
        print(f"[USING FUZZY MATCH] Song: {song_entity}, Artist: {artist_entity}")
    else:
        song_entity = clean_bert_output(entities.get("song"))
        artist_entity = clean_bert_output(entities.get("artist"))

    # Entity display
    song_match_info = f"Detected Song: **{song_entity if song_entity else 'N/A'}**"
    artist_match_info = f"Detected Artist: **{artist_entity if artist_entity else 'N/A'}**"
    mood_match_info = f"Detected Mood: **{mood_entity if mood_entity else 'N/A'}**"


    print(song_match_info)
    print(artist_match_info)
    print(mood_match_info)
    
    song_for_vec = input_song['Song'] if input_song is not None else song_entity
    artist_for_vec = input_song['Artist(s)'] if input_song is not None else artist_entity
    
    song_vec = get_song_vector(song_for_vec, artist_for_vec, embedder, df_song_info, song_embeddings, df_scaled_features)
    if mood_entity is not None and isinstance(mood_entity, str) and mood_entity.strip():
        emotion_vec = get_emotion_vector(mood_entity, embedder, scaled_emotion_means, emotion_labels)
    else:
        print("No mood provided, using neutral vector.")
        emotion_vec = np.zeros(scaled_emotion_means.shape[1])

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
        "score": 1.0 if input_song is not None else distance_to_similarity(distances[0][0]),
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
        
        radar_path = create_radar_chart(input_graph_vector, rec_vector, f"{rec_song_data['Song']}", features)
        
        similar_songs.append({
            "title": rec_song_data['Song'],
            "artist": rec_song_data['Artist(s)'],
            "score": distance_to_similarity(distances[0][np.where(top_indices == idx)[0][0]]),
            "radar_chart": radar_path
        })
        
        print(similar_songs)
        if len(similar_songs) == num_recommendations:
            break
            
    return {
        "main_song": main_song_display,
        "similar_songs": similar_songs,
        "song_match_info": f"Detected Song: **{song_for_vec if song_for_vec else 'N/A'}**",
        "artist_match_info": f"Detected Artist: **{artist_for_vec if artist_for_vec else 'N/A'}**",
        "mood_match_info": f"Detected Mood: **{mood_entity if mood_entity else 'N/A'}**"
    }