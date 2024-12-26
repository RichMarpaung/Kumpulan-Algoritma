import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Baca data
data = pd.read_csv('data/Costumer.csv').drop(columns=['IDPelanggan'])
data['Kelamin'] = data['Kelamin'].map({'Laki': 1, 'Perempuan': 0})

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Hitung Silhouette Score
silhouette_avg = silhouette_score(data_scaled, data['Cluster'])
print(f'Silhouette Score: {silhouette_avg:.4f}')

# Reduksi dimensi dengan PCA untuk visualisasi
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Plot hasil clustering dengan PCA
plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=data['Cluster'], cmap='viridis')
plt.title('K-Means Clustering with PCA (5 Klaster)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()
