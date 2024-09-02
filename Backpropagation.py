import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


data = pd.read_csv('data/data_balita_1k.csv')
data['Jenis Kelamin'] = data['Jenis Kelamin'].map({'laki-laki': 0, 'perempuan': 1})

#mekodekan status gizi severaly stunted(sangat stunting) = 0 , stunted = 1, normal = 2 , tinggi = 3
data['Status Gizi'] = data['Status Gizi'].map({'severely stunted': 0, 'stunted': 1, 'normal': 2, 'tinggi': 3}).astype(float)
# data['Umur (bulan)'] = data['Umur (bulan)'].astype(float)
#normalisasi
for i in range(len(data.columns)):
    data[data.columns[i]] = data[data.columns[i]].astype(float)

for i in range(len(data.columns)):
    for j in range(len(data)):
        maksimal = max(data[data.columns[i]])
        minimal = min(data[data.columns[i]])
        tamp = data.iloc[j][data.columns[i]] 
        data.loc[j,data.columns[i]] = (tamp-minimal) / (maksimal-minimal)



X = data.iloc[:, 1:6]
X = data.drop('Status Gizi', axis=1)
y = data['Status Gizi']
# membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# print(y_train)
#proses
# data_train = pd.concat([X_train, y_train], axis=1)
jmlh_input = X_train.shape[1]
jmlh_hidden = 6
jmlh_output = 1
alpha = 0.01
max_epoch = 1000   
convergen = False
target_error = 0.00001
#random bias
bias_hidden = np.random.rand(jmlh_hidden)
bias_target = np.random.randn()
#random bobot
bobot_V = np.random.rand(jmlh_input,jmlh_hidden)
bobot_W = np.random.rand(jmlh_hidden,jmlh_output)

epoch = 1
# print(bias_hidden)
while(epoch <= max_epoch and not convergen):
    z = np.zeros(jmlh_hidden)
    for i in range(len(X_train)):
        #forward
        for j in range(jmlh_hidden):
            sum = 0
            for k in range (jmlh_input):
                XkVj = bobot_V[k][j]*X_train.iloc[i][X_train.columns[k]]
                sum+=XkVj
            z_in = sigmoid(bias_hidden[j]+sum)
            z[j]=z_in
        
        y_in = 0
        for o in range(len(z)):
            WoZo = bobot_W[o]*z[o]
            y_in+=WoZo
        y = sigmoid(bias_target+y_in)
        #end forward
        
        if((abs(y_train.iloc[i]-y))<=target_error):
            # print(y)
            # print(y_train.iloc[i])
            # print(abs(y_train.iloc[i]-y))
            # print(abs(y_train.iloc[i]-y)<=target_error)
            convergen = True
            # print('hadir',i)
            break
      
        #back
        dot = (y_train.iloc[i]-y)*(y)*(1-y)
        delta_w = []

        for j in range(jmlh_hidden):
            delta_w.append(alpha*dot*z[j])

        delta_bias_w = alpha*dot

        dotz_in = []
       
        for o in range(jmlh_hidden):
            dot_in_i = dot[0]*bobot_W[o][0]
            dotz_in.append(dot_in_i)
        dot_z = []
        for p in range (jmlh_hidden):
            dot_z.append(dotz_in[p]*z[p]*(1-z[p]))
        

        delta_V = np.zeros((jmlh_input,jmlh_hidden))
        
        delta_bias_v = []
        for j in range (jmlh_hidden):
            for l in range (jmlh_input):
                
                delta_V[l][j] = alpha*dot_z[l]*X_train.iloc[j][X_train.columns[l]]
            delta_bias_v.append(alpha*dot_z[j])
        
        # update bobot dan bias
        for j in range (jmlh_hidden):
            for k in range(jmlh_input):
                #update bobot_v
                bobot_lama = bobot_V[k][j]
                bobot_V[k][j] = bobot_lama + delta_V[k][j]
            #update bias v
            bias_lama = bias_hidden[j]
            bias_hidden[j]=bias_lama+delta_bias_v[j]
            #update bobot w
            bobotW_lama = bobot_W[j]
            bobot_W[j] = bobot_lama+delta_w[j]
        biasW_lama = bias_target
        bias_target = biasW_lama+delta_bias_w
    epoch+=1
        
#pengujian 
y_predic = []
treshold = 0
for i in range (len(X_test)):
    z = np.zeros(jmlh_hidden)
    for j in range(jmlh_hidden):
        sum = 0
        for k in range (jmlh_input):
            XkVj = bobot_V[k][j]*X_test.iloc[i][X_test.columns[k]]
            sum+=XkVj
        z_in = sigmoid(bias_hidden[j]+sum)
        z[j]=(z_in)
    
    y_in = 0
    for o in range(len(z)):
        WoZo = bobot_W[o]*z[o]
        y_in+=WoZo
    y = sigmoid(bias_target+y_in)
    trashold = y_train.iloc[i]-y
    if(y>0 and y<= 0.25):
        y_predic.append(0)
    elif(y>0.25 and y<=0.5):
        y_predic.append(1)
    elif(y>0.5 and y<=0.75):
        y_predic.append(2)
    elif(y>0.75 and y<=1):
        y_predic.append(3)


for i in range (len(y_test)):
    y_test.iloc[i]*=3

print("Treshold : ",abs(trashold))
print("Akurasi : ",accuracy_score(y_test, y_predic))