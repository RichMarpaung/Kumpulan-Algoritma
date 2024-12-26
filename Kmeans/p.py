import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Mengimpor dataset
dataset = pd.read_csv('data/Costumer001.csv')

dataset['Kelamin'] = dataset['Kelamin'].map({'Laki': 1, 'Perempuan': 0})

for col in dataset.columns:
    maksimal = dataset[col].max()
    minimal = dataset[col].min()
    if maksimal != minimal:
        dataset[col] = (dataset[col] - minimal) / (maksimal - minimal)
    else:
        dataset[col] = 0.0

X = dataset.iloc[:, [1,2, 3, 4]].values

initial_centroids = np.array([
    [1,25,12,77],
    [1,58,15,88],
    [0,19,54,63]

    ])

# Fungsi untuk menampilkan hasil iterasi
def showIter(iter_num, kmeans, X):
    print(f"--- Iterasi {iter_num} ---")
    print("Centroid saat ini:")
    print(kmeans.cluster_centers_)
    # Hitung jumlah data di setiap klaster
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    for label, count in zip(labels, counts):
        print(f"Jumlah data di Klaster {label}: {count}")
    print()

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids, random_state=42, max_iter=1, n_init=1)
# kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, max_iter=1, n_init=1) # kalo mau random centroid

centroids_old = None
iter_num = 1

# Looping untuk mencetak informasi iterasi dan update centroid
while True:
    # Fit model dengan jumlah iterasi terbatas (max_iter=1)
    kmeans.fit(X)
    
    # Cetak informasi iterasi
    showIter(iter_num, kmeans, X)
    
    # Cek apakah centroid telah konvergen (tidak berubah)
    if centroids_old is not None and np.allclose(kmeans.cluster_centers_, centroids_old):
        print("Centroid telah konvergen, iterasi berhenti.")
        break
    
    # Simpan centroid saat ini sebagai centroid lama
    centroids_old = np.copy(kmeans.cluster_centers_)
    
    # Update iterasi
    iter_num += 1

    # Lanjutkan ke iterasi berikutnya dengan menggunakan centroid terbaru
    kmeans = KMeans(n_clusters=n_clusters, init=kmeans.cluster_centers_, random_state=42, max_iter=1, n_init=1)

# Centroid akhir
print("Centroid final:")
print(kmeans.cluster_centers_)

# # Visualisasi hasil clustering
# plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=50)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
# plt.title('K-Means Clustering dengan Centroid yang Diinisialisasi Manual')
# plt.xlabel('Fitur 1')
# plt.ylabel('Fitur 2')
# plt.legend()
# plt.show()