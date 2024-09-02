import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

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

# Menggunakan dataset dari CSV
data = pd.read_csv('input/Stunting_Dataset.csv')

# Mengubah kolom kategorikal menjadi numerik
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Breastfeeding'] = data['Breastfeeding'].map({'Yes': 1, 'No': 0})
data['Stunting'] = data['Stunting'].map({'Yes': 1, 'No': 0})

# Normalisasi data
for col in data.columns:
    maksimal = data[col].max()
    minimal = data[col].min()
    if maksimal != minimal:
        data[col] = (data[col] - minimal) / (maksimal - minimal)
    else:
        data[col] = 0.0

# Memisahkan kolom 'Stunting' untuk visualisasi

numerik = data.drop('Stunting', axis=1) 
k = 2
clusters, cluster_labels, final_centroids = kmeans_clustering(numerik, k)

print('Final centroids:\n', final_centroids)

data['Cluster'] = cluster_labels

for cluster_id in range(k):
    print(f"\nData points in Cluster {cluster_id}:")
    print(data[data['Cluster'] == cluster_id])

# Visualisasi kluster sesuai dengan status stunting

# Menggunakan 'Stunting' dan 'Age' untuk scatter plot
feature1 = 'Stunting'
feature2 = 'Age'

# Membuat scatter plot dengan Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x=feature1, y=feature2, hue='Cluster', palette='hsv', legend='full')

# Plotting centroids
# Gunakan kolom yang ada dalam final_centroids
centroid_feature1 = 'Age'
centroid_feature2 = 'Body Weight'
for i in range(k):
    plt.scatter(final_centroids.iloc[i][centroid_feature1], final_centroids.iloc[i][centroid_feature2], 
                s=400, c='black', marker='x')

plt.title('K-means Clustering sesuai dengan Stunting')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend(title='Cluster')
plt.show()
