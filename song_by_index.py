import pandas as pd
from data_utils import load_data

# Load data
df_meta = load_data("data/song_data.csv", index=True)
df_features = load_data("data/scaled_data.csv", index=True)

df_meta.index = df_meta.index.astype(int)
df_features.index = df_features.index.astype(int)

while True:
    song_index_input = input("Enter the index of the song (or 'q' to quit): ").strip()
    if song_index_input.lower() == 'q':
        break

    try:
        song_index = int(song_index_input)
    except ValueError:
        print("❌ Please enter a valid integer index.")
        continue

    if song_index not in df_meta.index:
        print("❌ That index is not found in the metadata.")
        continue

    # Show confirmation
    song_info = df_meta.loc[song_index]
    print(f"\n✅ Found: '{song_info['Song']}' by {song_info['Artist(s)']}")

    # Show vector
    song_vector = df_features.loc[song_index]
    print("\n🎯 Feature Vector:")
    print(song_vector.round(4).to_string())