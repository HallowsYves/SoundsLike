from data_utils import load_data


"""
    Find the vector of a song from its title
"""

df_features = load_data('data/scaled_data.csv', index=True)
df_song_info = load_data('data/song_data.csv', index=True)

assert df_song_info.index.equals(df_features.index), "Index mismatch!"

song_name = "Put You On"
matching = df_song_info[df_song_info["Song"] == song_name].index
artist_name = df_song_info.loc[matching[0],"Artist(s)"]

if matching.empty:
    print("Song not found")
else:
    song_index = matching[0]
    song_vector = [df_features.loc[song_index].values]

    print(f" Song: {song_name} | Artist: {artist_name}| Vector: {song_vector}")