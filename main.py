from sounds_like_utils import find_similar_songs

if __name__ == "__main__":
    prompt = input("Enter a song prompt (e.g., 'give me songs like Moon by Kanye West'): ")
    find_similar_songs(prompt)
