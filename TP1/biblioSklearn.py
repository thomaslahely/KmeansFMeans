from sklearn.cluster import KMeans
import numpy as np

# Générer des données synthétiques
points = np.array([
    [2, 10],  # A1
    [2, 5],   # A2
    [8, 4],   # A3
    [5, 8],   # B1
    [7, 5],   # B2
    [6, 4],   # B3
    [1, 2],   # C1
    [4, 9]    # C2
])
K = 3
initiale_centroides = np.array([
    [2, 10],
    [5, 8],
    [1, 2]
])
kmeans = KMeans(n_clusters=K, init=initiale_centroides, n_init=1)
kmeans.fit(points)
centroides_finaux = kmeans.cluster_centers_
print("Centroïdes finaux :")
for idx, centroide in enumerate(centroides_finaux):
    print(f"Centroïde {idx}: ({centroide[0]:.2f}, {centroide[1]:.2f})")
