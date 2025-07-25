import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
from data_utils import load_data

df_features = load_data("data/scaled_data.csv", index=True)
df_meta = load_data("data/song_data.csv", index=True)
scaled_emotion_means = np.load("data/scaled_emotion_means.npy")
with open("data/emotion_labels.txt", "r") as f:
    emotion_labels = [line.strip() for line in f]

exclude_emotions = {"thirst", "pink", "interest", "confusion", "angry", "True", "Love"}

if "Emotion" not in df_meta.columns:
    raise ValueError("Your song_data.csv needs an 'Emotion' column to do this analysis.")

df_labeled = df_meta[df_meta["Emotion"].notna() & ~df_meta["Emotion"].isin(exclude_emotions)]
df_labeled_features = df_features.loc[df_labeled.index]


pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_labeled_features)
df_labeled["PCA1"] = pca_result[:, 0]
df_labeled["PCA2"] = pca_result[:, 1]

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_labeled,
    x="PCA1", y="PCA2",
    hue="Emotion",
    palette="tab10",
    alpha=0.6,
    edgecolor=None
)

plt.title("PCA of Songs Colored by Emotion")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


