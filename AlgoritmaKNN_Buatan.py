import numpy as np 
import pandas as pd 
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

def prediksi (traning,test,k):
    y_predic = []


    for i in range (len(test)):
        kandidat = []
        for j in range(len(traning)):
            sum =0
            for z in range(len(test.columns)):
                tamp = (test.iloc[i][test.columns[z]] - traning.iloc[j][traning.columns[z]])**2
                sum+=tamp
            dx = np.sqrt(sum)
          
            if len(kandidat) < k:
                kandidat.append({"value":dx,"Status Gizi":traning.iloc[j]['Status Gizi']})
            else:
                for p in range (len(kandidat)):
                    if kandidat[p]["value"]>dx :
                        kandidat[p]["value"] = dx
                        kandidat[p]["Status Gizi"]= traning.iloc[j]['Status Gizi']
        status_gizi_values = [item["Status Gizi"] for item in kandidat]
        y_predic.append(Counter(status_gizi_values).most_common(1)[0][0])
    print(y_predic)

    return np.array(y_predic)



data = pd.read_csv('input/data_test.csv')
data['Jenis Kelamin'] = data['Jenis Kelamin'].map({'laki-laki': 0, 'perempuan': 1}) 

#mekodekan status gizi severaly stunted(sangat stunting) = 0 , stunted = 1, normal = 2 , tinggi = 3
# data['Status Gizi'] = data['Status Gizi'].map({'severely stunted': 0, 'stunted': 1, 'normal': 2, 'tinggi': 3})
data.sort_values('Tinggi Badan (cm)')

X = data.iloc[:, 1:6]
X = data.drop('Status Gizi', axis=1)
y = data['Status Gizi']

# membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55)

data_train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test])
# print(X_test)
y_p = prediksi(data_train,test,4)
y_df = pd.DataFrame(y_p, columns=['Status Gizi'])
test.reset_index(drop=True, inplace=True)
y_df.reset_index(drop=True, inplace=True)


# Menggabungkan DataFrame
result = pd.concat([test, y_df], axis=1)

# data_severely = result[result['Status Gizi'] == 'severely stunted']
# data_stunted = result[result['Status Gizi'] == 'stunted']
# data_normal = result[result['Status Gizi'] == 'normal']
# data_tinggi = result[result['Status Gizi'] == 'tinggi']
# print('Data Severely Stunted \n',data_severely)
# print('Data Stunted \n',data_stunted)
# print('Data Normal \n',data_normal)
# print('Data Tinggi \n',data_tinggi)
print("Hasil Prediksi\n",result)
accuracy = accuracy_score(y_test, y_p)
print(classification_report(y_test, y_p))

# print('confusion_matrix knn:')
print(confusion_matrix(y_test, y_p))
print('Akurasi model KNN:', accuracy)