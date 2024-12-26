import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Memuat dataset
file_path = '/mnt/data/water_potability.csv'
data = pd.read_csv(file_path)

# Menangani nilai yang hilang dengan mengisi menggunakan median dari setiap kolom
data = data.fillna(data.median())

# Memisahkan dataset menjadi fitur (X) dan target (y)
X = data.drop('Potability', axis=1)  # Fitur
y = data['Potability']  # Target klasifikasi

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Langkah 1: Menghitung probabilitas prior untuk setiap kelas
def hitung_prior(y):
    kelas = np.unique(y)  # Mendapatkan nilai unik pada kolom target (kelas 0 dan 1)
    priors = {c: np.mean(y == c) for c in kelas}  # Menghitung probabilitas prior untuk setiap kelas
    return priors

# Langkah 2: Menghitung mean dan standar deviasi untuk setiap fitur per kelas
def hitung_statistik(X, y):
    statistik = {}
    for c in np.unique(y):
        X_c = X[y == c]  # Memisahkan data berdasarkan kelas
        statistik[c] = {
            "mean": X_c.mean(axis=0),  # Menghitung rata-rata setiap fitur untuk kelas c
            "std": X_c.std(axis=0)     # Menghitung standar deviasi setiap fitur untuk kelas c
        }
    return statistik

# Langkah 3: Fungsi untuk menghitung probabilitas dengan distribusi Gaussian
def probabilitas_gaussian(x, mean, std):
    eksponen = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * eksponen

# Langkah 4: Menghitung posterior untuk setiap kelas
def hitung_posterior(X, priors, statistik):
    posterior = []
    for x in X:
        probabilitas_kelas = {}
        for c, prior in priors.items():
            # Memulai dengan nilai prior
            probabilitas_kelas[c] = np.log(prior)
            # Menambahkan log-likelihood untuk setiap fitur
            for i in range(len(x)):
                mean = statistik[c]["mean"][i]
                std = statistik[c]["std"][i]
                probabilitas_kelas[c] += np.log(probabilitas_gaussian(x[i], mean, std))
        posterior.append(probabilitas_kelas)
    return posterior

# Langkah 5: Membuat prediksi berdasarkan nilai posterior yang dihitung
def prediksi(X, priors, statistik):
    posterior = hitung_posterior(X, priors, statistik)
    prediksi = [max(p, key=p.get) for p in posterior]  # Memilih kelas dengan posterior tertinggi
    return prediksi

# Menerapkan langkah-langkah Naive Bayes manual pada dataset
priors = hitung_prior(y_train)  # Menghitung probabilitas prior
statistik = hitung_statistik(X_train.values, y_train.values)  # Menghitung statistik (mean dan std) untuk setiap fitur

# Membuat prediksi pada set pengujian
y_pred_manual = prediksi(X_test.values, priors, statistik)

# Menghitung akurasi
akurasi_manual = accuracy_score(y_test, y_pred_manual)
print("Akurasi Naive Bayes Manual:", akurasi_manual)
