import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
file_path = 'PRoduksi.csv'  # Sesuaikan nama file
data = pd.read_csv(file_path)

# Filter data untuk Majene
majene_data = data[data['Kabupaten dan Provinsi'].str.contains("Majene", case=False)]

# Konversi kolom numerik
numeric_columns = ['2015', '2016', '2018', '2020', '2021']
majene_data[numeric_columns] = majene_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Siapkan fitur (X) dan target (y)
X = majene_data[['2015', '2016', '2018', '2020']]
y = majene_data['2021']

# Split data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Buat model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi dan evaluasi
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output hasil
print("Prediksi:", y_pred)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R²):", r2)
import matplotlib.pyplot as plt

# Data Majene untuk plotting
majene_years = ['2015', '2016', '2018', '2020', '2021']
majene_types = majene_data['Jenis'].unique()

# Membuat plot
plt.figure(figsize=(10, 6))

# Loop untuk memplot data setiap jenis produksi
for jenis in majene_types:
    jenis_data = majene_data[majene_data['Jenis'] == jenis]
    plt.plot(majene_years, jenis_data[majene_years].values.flatten(), marker='o', label=jenis)

# Pengaturan tampilan grafik
plt.title('Tren Produksi di Majene (2015-2021)', fontsize=14)
plt.xlabel('Tahun', fontsize=12)
plt.ylabel('Hasil Produksi', fontsize=12)
plt.legend(title='Jenis Produksi')
plt.grid(True)
plt.tight_layout()

# Tampilkan grafik
plt.show()
