
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def generer_Points_donnees(n_samples=300, centers=None, cluster_std=1.0, random_state=42):
#nombre de points, centres des clusters, écart-type des clusters, graine aléatoire

#
    X, y=make_blobs(n_samples=n_samples, centers=centers,cluster_std=cluster_std, random_state=random_state)
#X : coordonnées des points, y : numéro du cluster de chaque point
    return X, y

centre_reels = [
    [0, 0],
    [5, 5],
    [10, 0]
]

X, y = generer_Points_donnees(n_samples=300, centers=centre_reels, cluster_std=10.0)

#nuage de points pour nos donnees 
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='plasma', s=50)
plt.title('Jeu de Données')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.show()
