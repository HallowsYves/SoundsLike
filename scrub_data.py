import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# * DEBUG REMOVE THIS LATER
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Load DataSet
try: 
    df = pd.read_csv('spotify_dataset.csv')
except FileNotFoundError:
    print("Could not find 'spotify_dataset.csv'.")


# Drop all duplicates
df.rename(columns={'song': 'Song'}, inplace=True)
df.drop_duplicates(subset=['Song', 'Artist(s)'], inplace=True)

# Drop all missing values
feature_cols = ['Popularity', 'Energy', 'Danceability', 'Positiveness']
df.dropna(subset=feature_cols, inplace=True)

df.reset_index(drop=True, inplace=True)

# Create Separate Df, to keep track of Song's And Artist
song_info = df[['Song', 'Artist(s)']].copy()



# Remove all non-numeric columns
columns_to_drop = [col for col in df.columns if col.startswith('Good for') or col.startswith('Similar') or col.startswith('Similarity')]

columns_to_drop.extend(['Song', 'Artist(s)', 'Genre', 'Album', 'text', 
                         'Length', 'Release Date', 'Key', 'Tempo', 'Loudness (db)', 
                         'Time signature', 'Explicit', 'emotion'])

df_numeric = df.drop(labels=columns_to_drop, axis=1)
print(df_numeric.columns)

# Scale
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_numeric)

# * Export Data
output_filename = 'spotify_for_clustering.csv'
df_numeric.to_csv(output_filename, index=False)

# Elbow Method
inertia_scores = []
k_values = range(2,11)

for k in k_values:
    k_means = KMeans(n_clusters=k, random_state=42, n_init=10)
    k_means.fit(scaled_features)
    inertia_scores.append(k_means.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia_scores, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_values)
plt.grid(True)
plt.show()
