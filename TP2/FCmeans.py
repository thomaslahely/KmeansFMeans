
import numpy as np
import skfuzzy as fuzz
from skfuzzy import cmeans
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
dunn_indices_fcmeans = []

for idx, centroides_init in enumerate(centroides_initiaux):
    print(f"\n=== Fuzzy C-Means === Scénario : {idx + 1} ===")
    print(f"Centres initiaux :\n{centroides_init}\n")
    #on crée une matrice de partition initiale qu'on initialise à 0
    u0 = np.zeros((K, X.shape[0]))
    #boucle pour chaque point de X
    for i in range(X.shape[0]):
    #on calcule la distance euclidienne entre le point i et les centroides initiaux
        distance = np.linalg.norm(X[i] - centroides_init, axis=1)
    #on inverse les distances pour avoir les degrés d'appartenance
        u0[:, i] = 1 / distance
    #on normalise les degrés d'appartenance
    u0 = u0 / np.sum(u0, axis=0)
#cntr : coordonnées des centres des clusters, u : matrice de partition finale
#u0 : matrice de partition initiale, d : distance euclidienne entre les points et les centroides
#jm : historique de la fonction objectif, p : nombre d'itérations qu'on a fait, fpc : coefficient de partition finale floue
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=X.T,# transposée de X car la fonction cmeans prend les données en colonnes
        c=K,# nombre de clusters
        m=2,# degré de flou
        error=0.005,#critère d'arrêt
        maxiter=10,#nombre d'itérations maximales
        init=u0,#matrice de partition initiale
    
    )
# on attribue à chaque point le label du cluster auquel il appartient le plus degre d'appartenance
    labels_fcmeans = np.argmax(u, axis=0)
    dunn_indice = dunn_index(X, labels_fcmeans)
    dunn_indices_fcmeans.append(dunn_indice)
    print(f"Indice de Dunn : {dunn_indice}")
    print("Centroides finaux:")
    print(cntr)


    plot.figure(figsize=(6, 4))
    plot.scatter(X[:, 0], X[:, 1], c=labels_fcmeans, cmap='viridis', s=50)
    plot.scatter(cntr[:, 0], cntr[:, 1], marker='X', s=200, c='red', label='Centroides')
    plot.title(f'Fuzzy C-Means - Scénario {idx + 1}')
    plot.xlabel('X1')
    plot.ylabel('X2')
    plot.legend()
    plot.grid(True)
    plot.show()
    