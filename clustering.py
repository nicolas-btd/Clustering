import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
from sklearn import metrics
from sklearn import cluster

def main():
    # Création du jeu de données de 4 gaussiennes (800 points par gaussienne)
    centers = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    x, y = skdata.make_blobs(n_samples=3200, centers=centers, cluster_std=0.5, random_state=42)

    # Affichage des données
    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis', marker='.')
    plt.title('donnees initiales')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('donnees_initiales.png')
    plt.show()

    # =========================================================================
    # 2. Trouver le nombre de classes
    # =========================================================================
    c_range = range(2, 11)
    sil = []
    J = []
    
    for k in c_range:
        km = cluster.KMeans(n_clusters=k, random_state=42, n_init=15)
        km.fit(x)
        
        sil = np.append(sil, metrics.silhouette_score(x, km.labels_))
        J = np.append(J, km.inertia_)

    figure = plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(c_range, J, '.-k')
    plt.xlabel("cluster number")
    plt.ylabel("J")
    
    plt.subplot(1, 2, 2)
    plt.plot(c_range, sil, '.-b')
    plt.xlabel("cluster number")
    plt.ylabel("silhouette index")
    
    plt.savefig('silhouette_et_J.png')
    plt.show()

if __name__ == '__main__':
    main()
