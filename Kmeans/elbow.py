
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('data/Costumer001.csv')

data = data.drop(columns=['IDPelanggan'])

data['Kelamin'] = data['Kelamin'].map({'Laki': 1, 'Perempuan': 0})

# for col in data.columns:
#     maksimal = data[col].max()
#     minimal = data[col].min()
#     if maksimal != minimal:
#         data[col] = (data[col] - minimal) / (maksimal - minimal)
#     else:
#         data[col] = 0.0


inertia = []
k_range = range(1, 11) 

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data)
    inertia.append(kmeans.inertia_)  

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_range)
plt.show()
