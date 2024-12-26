import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import time
def kmeans_clustering(data, k=2, max_iter=100):
    centroids = data.iloc[:k].astype(float)
    print('Initial centroids\n', centroids)

    cluster_labels = []
    iterations = 0
    converged = False

    while not converged and iterations < max_iter:
        cluster_labels = []
        
        for i in range(len(data)):
            distances = []
            for j in range(len(centroids)):
                distance = np.sqrt(((data.iloc[i] - centroids.iloc[j]) ** 2).sum())
                distances.append(distance)
            cluster_label = np.argmin(distances)
            cluster_labels.append(cluster_label)
        
        new_centroids = []
        for j in range(k):
            cluster_points = data[np.array(cluster_labels) == j]
            if not cluster_points.empty:
                new_centroid = cluster_points.mean()
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids.iloc[j])
        new_centroids = pd.DataFrame(new_centroids, columns=data.columns)
        
        if centroids.equals(new_centroids):
            converged = True
        else:
            centroids = new_centroids

        iterations += 1
        print(f'Iteration {iterations}\nCentroids\n{centroids}\n')

    result = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        result[f'C{label + 1}'].append(i + 1)
    
    return result, cluster_labels, centroids

data = pd.read_csv('data/Costumer.csv')

data['Kelamin'] = data['Kelamin'].map({'Laki': 1, 'Perempuan': 0})

for col in data.columns:
    maksimal = data[col].max()
    minimal = data[col].min()
    if maksimal != minimal:
        data[col] = (data[col] - minimal) / (maksimal - minimal)
    else:
        data[col] = 0.0

numerik = data.drop('IDPelanggan', axis=1) 

k = 3
start_time = time.time()
clusters, cluster_labels, final_centroids = kmeans_clustering(numerik, k)
end_time = time.time()  
execution_time = end_time - start_time
print('Final centroids:\n', final_centroids)

data['Cluster'] = cluster_labels
for cluster_id in range(k):
    print(f"\nData points in Cluster {cluster_id}:")
    print(data[data['Cluster'] == cluster_id])

print("Exectue ",execution_time)
