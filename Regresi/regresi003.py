import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

file_path = 'Produksi001.csv'  # Sesuaikan nama file
data = pd.read_csv(file_path)

# Daftar kabupaten dan tahun prediksi
kabupaten_list = data['Kabupaten'].unique()
future_years = [2025, 2026, 2027, 2028, 2030]

# Menyimpan hasil prediksi
all_predictions = []

# Loop untuk setiap kabupaten
for kabupaten in kabupaten_list:
    kabupaten_data = data[data['Kabupaten'] == kabupaten]
    jenis_list = kabupaten_data['Jenis'].unique()
    
    # Loop untuk setiap jenis produksi
    for jenis in jenis_list:
        jenis_data = kabupaten_data[kabupaten_data['Jenis'] == jenis]
        
        # Menentukan tahun yang ada di data
        available_years = [col for col in jenis_data.columns if col.isdigit()]
        
        # Buat fitur (tahun sebagai angka) dan target (produksi)
        years = np.array([int(year) for year in available_years]).reshape(-1, 1)
        production = jenis_data[available_years].values.flatten()
        
        # Melatih model regresi
        model = LinearRegression()
        model.fit(years, production)
        
        # Prediksi untuk tahun ke depan
        future_years_array = np.array(future_years).reshape(-1, 1)
        future_predictions = model.predict(future_years_array)
        all_predictions.append((kabupaten, jenis, future_years, future_predictions))
        
        # Visualisasi tren
        plt.plot(years.flatten(), production, marker='o', label=f'{jenis} (Data)')
        plt.plot(future_years, future_predictions, marker='x', linestyle='--', label=f'{jenis} (Prediksi)')
    
    # Pengaturan tampilan grafik untuk kabupaten
    plt.title(f'Prediksi Produksi di {kabupaten} (2025-2030)', fontsize=14)
    plt.xlabel('Tahun', fontsize=12)
    plt.ylabel('Hasil Produksi', fontsize=12)
    plt.legend(title='Jenis Produksi')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Menampilkan prediksi
for kabupaten, jenis, years, preds in all_predictions:
    print(f"Prediksi untuk {jenis} di {kabupaten}:")
    for year, pred in zip(years, preds):
        print(f"Tahun {year}: {pred:.2f}")
