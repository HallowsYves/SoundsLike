import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

try:
    df_scaled = pd.read_csv('data/scaled_features.csv')
    print('found dataset')
except FileNotFoundError:
    print("could not find dataset")

def elbow_method(data, max_k):
    means = []
    inertias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)
    
    fig =plt.subplots(figsize=(10, 5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Inertia')
    plt.grid(True)
    plt.show()


elbow_method(df_scaled, 10)

# Good range between 5 or 6