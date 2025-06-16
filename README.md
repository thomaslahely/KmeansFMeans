
# Implémentation Simple de K-Means en Python

Ce projet présente une implémentation basique de l’algorithme K-means pour la classification non supervisée sur un petit jeu de données 2D.

---

## Dataset

Le jeu de données est constitué de 9 points 2D :

```python
points = np.array([
    [2, 10],  # A1
    [2, 5],   # A2
    [8, 4],   # A3
    [5, 8],   # B1
    [7, 5],   # B2
    [6, 4],   # B3
    [1, 2],   # C1
    [4, 9],   # C2
    [5, 4]    # C3
])
````

Le nombre de clusters (`K`) est fixé à 3, avec des centroides initiaux choisis parmi les points :

```python
centroides_initiaux = np.array([
    [2, 10],  # A1
    [5, 8],   # B1
    [1, 2]    # C1
])
```

---

## Description des fonctions principales

### `euclidienne(x1, x2)`

Calcule la distance euclidienne entre deux points `x1` et `x2`.

---

### `assigner_point_a_clusters(centroides, points)`

Attribue chaque point au cluster dont le centroïde est le plus proche en calculant les distances euclidiennes.

---

### `mettre_a_jour_centroides(clusters)`

Met à jour les centroides en calculant la moyenne des points appartenant à chaque cluster.

---

### `voir_changement(anciens_centroides, nouveaux_centroides)`

Détecte la convergence : retourne `True` si les centroides n’ont pas bougé, `False` sinon.

---

### `kmeans(points, centroides_initiaux)`

Fonction principale qui réalise les itérations jusqu’à convergence :

* Assigne les points aux clusters.
* Met à jour les centroides.
* Affiche les informations à chaque itération.
* Renvoie les centroides finaux, les clusters, et le nombre d’itérations.

---

## Résultat attendu

Le programme affiche à chaque itération :

* Les coordonnées des centroides.
* Les points de chaque cluster.

Après convergence, il affiche les centroides finaux et produit une visualisation graphique des clusters et de leurs centroides.

---

## Installation

Installer les dépendances Python :

```bash
pip install numpy matplotlib
```

---

## Utilisation

Lancer le script Python. La visualisation s’ouvre automatiquement à la fin du traitement.

---

## Exemple de sortie

```
Itération 1:
Centroïde 0: (2.00, 7.67)
Cluster 0: [array([2, 10]), array([2, 5]), array([4, 9])]
Centroïde 1: (6.40, 5.20)
Cluster 1: [array([8, 4]), array([7, 5]), array([6, 4]), array([5, 4])]
Centroïde 2: (1.00, 2.00)
Cluster 2: [array([1, 2])]
...
Centroides finaux après X itérations :
Centroïde 0: (2.00, 7.67)
Centroïde 1: (6.40, 5.20)
Centroïde 2: (1.00, 2.00)
```

---

## Notes

* Cet algorithme est sensible au choix initial des centroides.
* Il peut être adapté pour des données multidimensionnelles.
* Pour des applications plus robustes, utiliser les fonctions KMeans de bibliothèques comme scikit-learn.

---

## Auteur

Ton Nom - Projet d’apprentissage Python & Machine Learning

---

N’hésite pas si tu souhaites que je te prépare un fichier README pour les autres versions (Fuzzy C-Means, visualisation plus avancée, etc.) !
