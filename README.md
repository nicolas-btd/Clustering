# Exploration des algorithmes de Clustering

Ce dépôt vise à explorer et comparer différents algorithmes de clustering en Machine Learning.

## Avancement

### 1. Génération des données initiales
Pour commencer notre exploration, nous avons créé un jeu de données synthétique composé de quatre gaussiennes en 2D. Chaque gaussienne est centrée sur des coordonnées distinctes ((-1, -1), (-1, 1), (1, -1) et (1, 1)) et possède un écart-type de 0.5. Nous avons généré 800 points par gaussienne, pour un total de 3200 points.

![Données initiales](donnees_initiales.png)

- [x] Génération de données synthétiques (4 gaussiennes, 800 points chacune).

### 2. Trouver le nombre de classes optimal
Nous avons utilisé la méthode du coude sur l'inertie ($J$) et l'indice de silhouette pour déterminer le nombre idéal de clusters sur notre jeu de données (k allant de 2 à 10).

![Inertie et Silhouette](silhouette_et_J.png)

- [x] Trouver le nombre de classes optimal via K-Means (Inertie et Silhouette).

### 3. Évolution de l'algorithme K-Means
En fixant un nombre de clusters à 4 et une initialisation spécifique des centroïdes, nous observons le déplacement des centres et la diminution de l'inertie sur les 4 premières itérations.

![Évolution K-Means](iterations_kmeans.png)

- [x] Évolution de la fonction objectif à chaque itération.

### 4. Évaluation supervisée avec ARI
Nous utilisons la métrique de validation supervisée Adjusted Rand Index (ARI) pour évaluer la qualité de notre partition finale (en la comparant avec les vraies étiquettes du jeu de données). L'ARI obtenu permet de valider le bon fonctionnement de K-Means.

- [x] Calcul de l'Adjusted Rand Index (ARI).

### 5. Comparaison des algorithmes

Nous avons commencé par évaluer **KMeans** de manière automatisée sur différents jeux de données générés avec Scikit-Learn : Gaussiennes, Lunes (Moons), Données uniformes, Gaussiennes non équilibrées et Cercles.
Nous avons ensuite ajouté **AgglomerativeClustering**.

| Dataset | KMeans | Agglomerative |
|---|---|---|
| Gaussiennes | 0.8901 | 0.8296 |
| Moons | 0.3049 | 0.2930 |
| Uniform | 0.0003 | 0.0015 |
| Unbalanced | 0.6776 | 0.5789 |
| Circles | 0.5825 | 0.5865 |
