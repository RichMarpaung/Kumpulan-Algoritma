import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import time
data = pd.read_csv('data/Costumer.csv')
initial_centroids = np.array([
        [0,23,87,29],
        [1,60,4,30],
        [0,21,73,30],
    ])
data = data.drop(columns=['IDPelanggan'])

data['Kelamin'] = data['Kelamin'].map({'Laki': 1, 'Perempuan': 0})
start_time = time.time() 
kmeans = KMeans(n_clusters=3, init=initial_centroids,random_state=42)
data['Cluster'] = kmeans.fit_predict(data)

end_time = time.time()  
execution_time = end_time - start_time
print(execution_time)
centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=['Kelamin', 'Usia', 'Rating_belanja (1-100)', 'Pendapatan (juta Rp)'])

print("Clustered Data:\n", data.head())
print("\nFinal Centroids:\n", centroids_df)
