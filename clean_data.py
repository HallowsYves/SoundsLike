import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    df = pd.read_csv('data/spotify_dataset.csv')
    print('found dataset')
except FileNotFoundError:
    print("could not find dataset")

# Clean up the data: drop duplicates, rename

df.rename(columns={'song': 'Song', 'emotion':'Emotion'}, inplace=True)
df.drop_duplicates(subset=['Song', 'Artist(s)'], inplace=True)

keep_columns = ['Artist(s)', 'Song', 'Emotion', 'Positiveness', 'Danceability','Genre', 'Energy', 'Popularity']
df = df[keep_columns]

# Drops missing values/duplicates
df.dropna(subset=keep_columns, inplace=True)
df.drop_duplicates(subset=['Song', 'Artist(s)'], inplace=True)

df.reset_index(drop=True, inplace=True)

# Info with the Song/Artist
song_info = df[['Song', 'Artist(s)']].copy()

# Remove non-numeric columns

print(df.columns.tolist())
non_numeric_cols = ['Song', 'Artist(s)', 'Emotion', 'Genre']
df_numeric = df.drop(columns=non_numeric_cols, axis=1, errors='ignore')
print(df_numeric.columns.tolist())

# Trying Standard Deviation

scaler = StandardScaler()
df[['Positiveness_T', 'Danceability_T', 'Energy_T', 'Popularity_T']] = scaler.fit_transform(df[['Positiveness', 'Danceability', 'Energy', 'Popularity']])

# Saving Scaled Data

df_scaled = df[['Positiveness_T', 'Danceability_T', 'Energy_T', 'Popularity_T']]
df_scaled.to_csv('data/scaled_features.csv', index=False)