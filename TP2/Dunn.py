
import numpy as np


def dunn_index(X, labels):
    from scipy.spatial.distance import pdist, cdist, squareform
    clusters = [X[labels == i] for i in np.unique(labels)]
    # Calcul des distances inter-clusters
    delta = np.min([cdist(u, v).min()
                    for i, u in enumerate(clusters) for v in clusters[i + 1:]])
    # Calcul des diamètres des clusters
    diameters = np.array([pdist(cluster).max() for cluster in clusters])
    return delta / diameters.max()
