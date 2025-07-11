import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from data_utils import load_data

df_scaled = load_data('data/scaled_data.csv', index=True)

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