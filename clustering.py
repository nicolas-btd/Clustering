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

if __name__ == '__main__':
    main()
