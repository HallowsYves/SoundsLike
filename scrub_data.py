import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data_utils import load_and_clean_data

# * DEBUG REMOVE THIS LATER
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


df_numeric, song_info = load_and_clean_data()

if df_numeric is not None:
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
