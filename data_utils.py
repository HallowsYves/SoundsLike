import pandas as pd

def load_and_clean_data(filepath='spotify_dataset.csv'):
    try:
        df = pd.read_csv(filepath)
    except:
        print(f"ERROR: File not found: {filepath}")
    
    df.rename(columns={'song': 'Song'}, inplace=True)
    
    # Drop all missing values
    feature_cols = ['Popularity', 'Energy', 'Danceability', 'Positiveness']
    df.dropna(subset=feature_cols, inplace=True)

    df.drop_duplicates(subset=['Song', 'Artist(s)'], inplace=True, keep='first')


    df.reset_index(drop=True, inplace=True)

    df_numeric = df[feature_cols].copy()
    song_info = df[['Song', 'Artist(s)']].copy()

    print("Data loaded and cleaned.")
    return df_numeric, song_info