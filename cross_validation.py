import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

# we donts need any of these because we are finding similarities not predicting
try:
    df = pd.read_csv('data/your_full_dataset.csv')  
    print('Found dataset')
except FileNotFoundError:
    print("Could not find dataset")


X = df[['Positiveness', 'Danceability', 'Energy', 'Popularity']]
y = df['Emotion'] 

def cross_validation(X, y, max_k):
    k_values = list(range(1, max_k))
    scores = []

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X_scaled, y, cv=5)
        scores.append(np.mean(score))
    
    plt.plot(k_values, scores, 'o-')
    plt.xlabel("K Values")
    plt.ylabel("Accuracy Score")
    plt.title("KNN Cross-Validation")
    plt.grid(True)
    plt.show()

cross_validation(X, y, max_k=30)