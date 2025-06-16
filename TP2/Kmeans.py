from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plot
from JeuDeDonnees import X, y
from Dunn import dunn_index


K = 3
centroides_initiaux = [
    # Scénario 1 : Centres initiaux proches des centres réels
    np.array([
        [0, 0],
        [5, 5],
        [10, 0]
    ]),
    # Scénario 2 : Centres initiaux éloignés des centres réels
    np.array([
        [2, 2],
        [6, 6],
        [8, -2]
    ]),
    # Scénario 3 : Centres initiaux aléatoires
    # on creé un génerateur de nombre aléatoire avec une graine fixée pour permettre la reproduction des résultats
    #on génere des point aléatoires au plus bas -5 et au plus haut 15 qui ont la meme probabilité d'apparaitre
    #on crée un tableau de 3 lignes et 2 colonnes car notre K=3
    np.random.RandomState(42).uniform(low=-5, high=15, size=(K, 2))
]
dunn_indices_kmeans = []

for idx, centroides_init in enumerate(centroides_initiaux):
    print(f"\n=== K-Means - Scénario {idx + 1} ===")
    print(f"Centres initiaux :\n{centroides_init}\n")
    kmeans = KMeans(n_clusters=K, init=centroides_init, n_init=1, max_iter=10, random_state=42)
    kmeans.fit(X)
    centroides_finaux = kmeans.cluster_centers_
    labels_kmeans = kmeans.labels_
    dunn_indice = dunn_index(X, labels_kmeans)
    dunn_indices_kmeans.append(dunn_indice)
    print(f"Indice de Dunn : {dunn_indice}")
    
    print("Centroides finaux:")
    print(centroides_finaux)

    plot.figure(figsize=(6, 4))
    plot.scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis', s=50)
    plot.scatter(centroides_finaux[:, 0], centroides_finaux[:, 1], marker='X', s=200, c='red', label='Centroides')
    plot.title(f'K-Means - Scénario {idx + 1}')
    plot.xlabel('X1')
    plot.ylabel('X2')
    plot.legend()
    plot.grid(True)
    plot.show()