import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
from sklearn import metrics
from sklearn import cluster

def main():
    # Create the 4 Gaussians dataset (800 points per Gaussian)
    centers = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    x, y = skdata.make_blobs(n_samples=3200, centers=centers, cluster_std=0.5, random_state=42)

    # Plot initial data
    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis', marker='.')
    plt.title('initial data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('initial_data.png')
    plt.show()

    # =========================================================================
    # 2. Finding the optimal number of clusters
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
    plt.ylabel("J (Inertia)")
    
    plt.subplot(1, 2, 2)
    plt.plot(c_range, sil, '.-b')
    plt.xlabel("cluster number")
    plt.ylabel("silhouette index")
    
    plt.savefig('silhouette_and_inertia.png')
    plt.show()

    # =========================================================================
    # 3. Evolution of the objective function at each iteration
    # =========================================================================
    c = 4
    figure = plt.figure(figsize=(16, 3))
    centroids = np.array([[-2, -1.5], [-2, -1], [-2, 0], [1.5, 1]])
    
    for i in range(1, 5):
        res = cluster.KMeans(n_clusters=c, n_init=1, init=centroids, max_iter=i, random_state=42).fit(x)
        
        plt.subplot(1, 4, i)
        plt.scatter(x[:, 0], x[:, 1], c=res.labels_, cmap='viridis', marker='.')
        plt.plot(res.cluster_centers_[:, 0], res.cluster_centers_[:, 1], 'xr', markersize=12, markeredgewidth=4)
        plt.title(f"iteration={i}, inertia={int(res.inertia_)}")

    plt.tight_layout()
    plt.savefig('kmeans_iterations.png')
    plt.show()

    # =========================================================================
    # 4. Supervised evaluation with ARI (Adjusted Rand Index)
    # =========================================================================
    res = cluster.KMeans(n_clusters=4, random_state=42, n_init=10).fit(x)
    ARI = np.abs(metrics.adjusted_rand_score(y, res.labels_))
    print(f"Adjusted Rand Index (ARI) = {ARI:.4f}")

    # =========================================================================
    # 5. Algorithm comparison
    # =========================================================================
    compare_algorithms()

def runKmeans(x, y, cmin=2, cmax=10):
    best_sil = -1
    best_k = cmin
    for k in range(cmin, cmax + 1):
        km = cluster.KMeans(n_clusters=k, random_state=42, n_init=10).fit(x)
        sil = metrics.silhouette_score(x, km.labels_)
        if sil > best_sil:
            best_sil = sil
            best_k = k
    
    km = cluster.KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(x)
    ari = metrics.adjusted_rand_score(y, km.labels_)
    return np.abs(ari), best_k

def runAgglomerative(x, y, cmin=2, cmax=10):
    best_sil = -1
    best_k = cmin
    for k in range(cmin, cmax + 1):
        agg = cluster.AgglomerativeClustering(n_clusters=k).fit(x)
        sil = metrics.silhouette_score(x, agg.labels_)
        if sil > best_sil:
            best_sil = sil
            best_k = k
            
    agg = cluster.AgglomerativeClustering(n_clusters=best_k).fit(x)
    ari = metrics.adjusted_rand_score(y, agg.labels_)
    return np.abs(ari), best_k

def runSpectral(x, y, cmin=2, cmax=10):
    import warnings
    warnings.filterwarnings('ignore') # SpectralClustering can be noisy
    best_sil = -1
    best_k = cmin
    for k in range(cmin, cmax + 1):
        try:
            sp = cluster.SpectralClustering(n_clusters=k, random_state=42, assign_labels='kmeans').fit(x)
            if len(set(sp.labels_)) > 1:
                sil = metrics.silhouette_score(x, sp.labels_)
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
        except:
            pass
            
    sp = cluster.SpectralClustering(n_clusters=best_k, random_state=42, assign_labels='kmeans').fit(x)
    ari = metrics.adjusted_rand_score(y, sp.labels_)
    return np.abs(ari), best_k

def runDBSCAN(x, y):
    best_sil = -1
    best_eps = 0.5
    best_min_samples = 5
    
    for eps in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
        for min_samples in [3, 5, 10]:
            db = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(x)
            if len(set(db.labels_)) > 1:
                sil = metrics.silhouette_score(x, db.labels_)
                if sil > best_sil:
                    best_sil = sil
                    best_eps = eps
                    best_min_samples = min_samples
                    
    db = cluster.DBSCAN(eps=best_eps, min_samples=best_min_samples).fit(x)
    ari = metrics.adjusted_rand_score(y, db.labels_)
    return np.abs(ari), best_eps

def runMeanShift(x, y):
    bandwidth = cluster.estimate_bandwidth(x, quantile=0.2)
    if bandwidth == 0:
        bandwidth = 1.0
    ms = cluster.MeanShift(bandwidth=bandwidth).fit(x)
    ari = metrics.adjusted_rand_score(y, ms.labels_)
    return np.abs(ari), len(set(ms.labels_))

def generate_datasets():
    datasets = {}
    
    # 4 initial Gaussians
    centers = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    datasets['Gaussians'] = skdata.make_blobs(n_samples=400, centers=centers, cluster_std=0.5, random_state=42)
    
    # Moons
    datasets['Moons'] = skdata.make_moons(n_samples=400, noise=0.1, random_state=42)
    
    # Uniform data
    x_uni = np.round(np.random.rand(200, 2), 2)
    y_uni = np.random.randint(0, 3, 200)
    datasets['Uniform'] = (x_uni, y_uni)
    
    # Unbalanced groups
    datasets['Unbalanced'] = skdata.make_blobs(n_samples=[100, 200, 50], 
                                              centers=[[-1, -1], [0, 0], [1, 1]], 
                                              cluster_std=[0.1, 0.5, 0.1], 
                                              random_state=42)
    
    # Circles
    datasets['Circles'] = skdata.make_circles(n_samples=400, factor=0.2, noise=0.1, random_state=42)
    
    return datasets

def compare_algorithms():
    print("\n--- Clustering Algorithms Comparison ---")
    datasets = generate_datasets()
    
    results = {
        'Dataset': [],
        'KMeans': [],
        'Agglomerative': [],
        'Spectral': [],
        'DBSCAN': [],
        'MeanShift': []
    }
    
    for name, (x, y) in datasets.items():
        print(f"Evaluating dataset: {name}")
        results['Dataset'].append(name)
        
        # KMeans
        ari, _ = runKmeans(x, y)
        results['KMeans'].append(round(ari, 4))
        
        # Agglomerative
        ari, _ = runAgglomerative(x, y)
        results['Agglomerative'].append(round(ari, 4))
        
        # Spectral
        ari, _ = runSpectral(x, y)
        results['Spectral'].append(round(ari, 4))
        
        # DBSCAN
        ari, _ = runDBSCAN(x, y)
        results['DBSCAN'].append(round(ari, 4))
        
        # MeanShift
        ari, _ = runMeanShift(x, y)
        results['MeanShift'].append(round(ari, 4))
        
    import csv
    
    print("\nARI Scores Table:")
    headers = ["Dataset", "KMeans", "Agglomerative", "Spectral", "DBSCAN", "MeanShift"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    
    for i in range(len(results['Dataset'])):
        row = [str(results[col][i]) for col in headers]
        print("| " + " | ".join(row) + " |")
        
    with open('clustering_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(len(results['Dataset'])):
            writer.writerow([results[col][i] for col in headers])
            
    print("\nResults saved to 'clustering_results.csv'")

if __name__ == '__main__':
    main()
