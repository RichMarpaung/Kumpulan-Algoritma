import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import os

#PROSES EDA
#membaca data set
data = pd.read_csv('input/data_balita_besar.csv')


print(data.isna().sum(),"\n") #pengecekan data null

#PROSES KNN

#mengkodekan jenis kelamin laki = 0 perempuan = 1
data['Jenis Kelamin'] = data['Jenis Kelamin'].map({'laki-laki': 0, 'perempuan': 1}) 

#mekodekan status gizi severaly stunted(sangat stunting) = 0 , stunted = 1, normal = 2 , tinggi = 3
# data['Status Gizi'] = data['Status Gizi'].map({'severely stunted': 0, 'stunted': 1, 'normal': 2, 'tinggi': 3})


X = data.iloc[:, 1:6]
X = data.drop('Status Gizi', axis=1)
y = data['Status Gizi']

# membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



knn = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=7)
#melakukan latihan model KNN dengan data latih
knn.fit(X_train, y_train)
# membuat prediksi dengan data uji
y_pred = knn.predict(X_test)

# menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print('Akurasi model KNN:', accuracy)



test = pd.DataFrame({'Umur (bulan)': [0], 'Jenis Kelamin': [1], 'Tinggi Badan (cm)': [41.2]})
pred = knn.predict(test)

data.columns
X_test.reset_index(drop=True, inplace=True)
y_df = pd.DataFrame(y_pred, columns=['Status Gizi'])
y_df.reset_index(drop=True, inplace=True)
# Gabungkan X_train dan y_train_array ke dalam satu array
data_predict = pd.concat([X_test, y_df], axis=1)
print(data_predict)
print(type(y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
print('confusion_matrix knn:')
print(confusion_matrix(y_test, y_pred))
print('Akurasi model KNN:', accuracy)