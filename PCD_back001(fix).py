import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Memuat data

data = pd.read_csv('data/detected_colors100.csv')
data['label'] = data['label'].map({'Tidak Lengkap': 0, 'Lengkap': 1})

for i in range(len(data.columns)):
    data[data.columns[i]] = data[data.columns[i]].astype(float)

for i in range(len(data.columns)):
    for j in range(len(data)):
        maksimal = max(data[data.columns[i]])
        minimal = min(data[data.columns[i]])
        tamp = data.iloc[j][data.columns[i]] 
        data.loc[j,data.columns[i]] = (tamp-minimal) / (maksimal-minimal)


# Pisahkan fitur (X) dan target (y)
# Ubah 'target_column' menjadi nama kolom target dalam data Anda

X = data.drop(columns=['label'])
y = data['label']

# Memisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=50)

# Standarisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Membangun model Neural Network (MLPClassifier)
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# Melatih model
model.fit(X_train, y_train)

# Prediksi pada data uji
y_pred = model.predict(X_test)

# Evaluasi model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the distribution of RGB values for each class
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_, 
            annot_kws={"size": 20})  # Menambahkan parameter annot_kws
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.show()