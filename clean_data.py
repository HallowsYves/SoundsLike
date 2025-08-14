from data_utils import clean_data, scale_data

"""Cleans, splits, and scales the original dataset

    Cleans the spotify dataset to keep only 11 columns.
    Then it's split into 2 dataframes: numeric and alphabetic.
    It only scales the numeric values for the KNN graph. 
"""

full_clean = clean_data(
    filepath='data/spotify_dataset.csv',
    rename={'song': 'Song', 'emotion': 'Emotion'},
    duplicates=['Song', 'Artist(s)'],
    keep=[
        'Artist(s)', 'Song', 'Emotion', 'Genre',
        'Positiveness', 'Danceability', 'Energy', 'Popularity',
        'Liveness', 'Acousticness', 'Instrumentalness'
    ],
    save_path='data/clean_data.csv'
)

df_numeric = full_clean[['Positiveness', 'Danceability', 'Energy', 'Popularity',
        'Liveness', 'Acousticness', 'Instrumentalness']].copy()
df_numeric.to_csv('data/numeric_data.csv')

df_song = full_clean[['Artist(s)', 'Song', 'Emotion', 'Genre']].copy()
df_song.to_csv('data/song_data.csv', index=True)

scale_data('data/numeric_data.csv',
           index=True, 
           save_path='data/scaled_data.csv'
)