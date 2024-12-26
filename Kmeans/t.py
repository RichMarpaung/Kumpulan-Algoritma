import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import time
import matplotlib.pyplot as plt

# Baca data
data = pd.read_csv('data/Costumer.csv')

# Inisialisasi centroid awal (4 centroid)
initial_centroids = np.array([
    [0, 23, 87, 29],
    [1, 60, 4, 30],
    [0, 21, 73, 30],
    [1, 45, 50, 25]  # Tambahan centroid ke-4
])

# Hapus kolom ID pelanggan
data = data.drop(columns=['IDPelanggan'])

# Ubah kolom 'Kelamin' menjadi numerik
data['Kelamin'] = data['Kelamin'].map({'Laki': 1, 'Perempuan': 0})

# Normalisasi data menggunakan Min-Max Scaling
# for col in data.columns:
#     maksimal = data[col].max()
#     minimal = data[col].min()
#     if maksimal != minimal:
#         data[col] = (data[col] - minimal) / (maksimal - minimal)
#     else:
#         data[col] = 0.0

# Mulai menghitung waktu eksekusi
start_time = time.time()

# K-Means clustering dengan 4 klaster
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data)

# Akhiri waktu eksekusi
end_time = time.time()
execution_time = end_time - start_time

# Hitung silhouette score
silhouette_avg = silhouette_score(data.drop(columns=['Cluster']), data['Cluster'])

# Tampilkan hasil centroid
centroids_df = pd.DataFrame(
    kmeans.cluster_centers_, 
    columns=['Kelamin', 'Usia', 'Rating_belanja (1-100)', 'Pendapatan (juta Rp)']
)

print(f"Execution Time: {execution_time:.4f} seconds")
print(f"Silhouette Score: {silhouette_avg:.4f}")
print("\nClustered Data:\n", data.head())
print("\nFinal Centroids:\n", centroids_df)

# === Visualisasi Scatter Plot ===
plt.figure(figsize=(8, 6))

# Plot data klaster berdasarkan dua fitur: Usia dan Pendapatan
for cluster in np.unique(data['Cluster']):
    plt.scatter(
        data[data['Cluster'] == cluster]['Usia'], 
        data[data['Cluster'] == cluster]['Pendapatan (juta Rp)'],
        label=f'Cluster {cluster}'
    )

# Plot centroid
plt.scatter(
    kmeans.cluster_centers_[:, 1],  # Usia (fitur ke-2)
    kmeans.cluster_centers_[:, 3],  # Pendapatan (fitur ke-4)
    s=200, c='red', marker='X', label='Centroids'
)

# Konfigurasi plot
plt.title('K-Means Clustering: Usia vs Pendapatan (4 Klaster)')
plt.xlabel('Usia (Scaled)')
plt.ylabel('Pendapatan (juta Rp) (Scaled)')
plt.legend()
plt.grid(True)
plt.show()
