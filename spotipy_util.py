import spotipy
import os
import sys
import streamlit as st

from spotipy.oauth2 import SpotifyClientCredentials

from dotenv import load_dotenv

load_dotenv(".env")


client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")


def init_spotify():
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=st.secrets['spotify']['CLIENT_ID'],
        client_secret=st.secrets['spotify']['CLIENT_SECRET']
    ))


def get_spotify_track(spotify, artist, song):
    query = f"{song} {artist}"
    result = spotify.search(q=query, type="track", limit=1)

    if result and result['tracks']['items']:
        return result['tracks']['items'][0]
    return None