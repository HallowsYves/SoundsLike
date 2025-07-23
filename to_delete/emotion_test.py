import pandas as pd
from data_utils import load_data

df = load_data("data/clean_data.csv", index=True)

features = ["Positiveness", "Danceability", "Energy", "Popularity", "Liveness", "Acousticness", "Instrumentalness"]
emotion_vectors = df.groupby("Emotion")[features].mean()
mood_vector = emotion_vectors.loc["surprise"].values
print(mood_vector)
# joy - [49.1956159  59.04124288 62.12736711 30.73534848 20.03502213 27.06547638 6.93784523]
# sadness - [43.76780032 55.10455395 62.37330128 30.42997532 19.21053001 27.02631227 8.33680465]
# anger - [48.38930722 62.7325758  65.77659318 29.89105095 20.52344347 20.84233126 6.42403519]
# fear - [45.11223077 56.187      63.355      30.54584615 19.63838462 25.65938462 8.58542308]
# love - [48.51018559 57.82138776 57.82702234 31.13483589 19.02691201 32.07845069 6.8412467 ]
# surprise - [47.60172899 57.76176116 61.58906313 30.60735826 19.07076799 28.52935263 7.02171291]