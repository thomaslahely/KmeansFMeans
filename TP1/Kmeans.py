import numpy as np
import matplotlib.pyplot as plt 
points = np.array([
    [2, 10],  #A1
    [2, 5],   #A2
    [8, 4],   #A3
    [5, 8],   #B1
    [7, 5],   #B2
    [6, 4],   #B3
    [1, 2],   #C1
    [4, 9],   #C2
    [5,4]     #C3
])
centroides_initiaux = np.array([
    [2, 10],  # A1
    [5, 8],   # B1
    [1, 2]    # C1
])
K =3


def euclidienne(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def assigner_point_a_clusters(centroides, points):
    clusters = {}
    for i in range(K):
        clusters[i] = []
    for point in points:
        distances = [euclidienne(point, centroide) for centroide in centroides]
        centroide_proche = np.argmin(distances)
        clusters[centroide_proche].append(point)
    return clusters


def mettre_a_jour_centroides(clusters):
    centroides = []
    for i in range(K):
        moyenne_cluster = np.mean(clusters[i], axis=0)
        centroides.append(moyenne_cluster)
    return centroides   

def voir_changement(anciens_centroides, nouveaux_centroides):
    distances = [euclidienne(ancien, nouveau) for ancien, nouveau in zip(anciens_centroides, nouveaux_centroides)]
    return sum(distances) == 0

def kmeans(points,centroides_initiaux):
    centroides = centroides_initiaux
    iterations = 0
    while True:
        anciens_centroides = centroides
        clusters = assigner_point_a_clusters(centroides, points)
        centroides = mettre_a_jour_centroides(clusters)
        iterations += 1

        print(f"\nItération {iterations}:")
        for idx, centroide in enumerate(centroides):
            print(f"Centroïde {idx}: ({centroide[0]:.2f}, {centroide[1]:.2f})")
        for idx in clusters:
            print(f"Cluster {idx}: {clusters[idx]}")
        
        if voir_changement(anciens_centroides, centroides):
            break
    return centroides, clusters, iterations



Assignation_finale, clusters, iterations = kmeans(points, centroides_initiaux)
print(f"Centroides finaux après {iterations} itérations :")
for index, centroide in enumerate(Assignation_finale):
    print(f"Centroïde {index}: ({centroide[0]:.2f}, {centroide[1]:.2f})")
colors = ['r', 'g', 'b']  
plt.figure()
for index in clusters:
    cluster_points = np.array(clusters[index])
    if cluster_points.size > 0:
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[index], label=f'Cluster {index}')
centroides_array = np.array(Assignation_finale)
plt.scatter(centroides_array[:, 0], centroides_array[:, 1], c='k', marker='x', s=100, label='Centroides')
plt.title(f'Clusters finaux après {iterations} itérations')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()