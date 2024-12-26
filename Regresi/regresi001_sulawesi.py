import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
file_path = 'Produksi002.csv'  # Sesuaikan nama file
data = pd.read_csv(file_path)


kabupaten_data = data[data['Kabupaten'].str.contains("Sulawesi Barat", case=False)]

kabupaten_tahun = ['2015', '2016', '2018', '2020', '2021']
type = kabupaten_data['Jenis'].unique()

tahun_prediksi = [2022,2023,2024,2025, 2026,]
predictions = []

for jenis in type:
    jenis_data = kabupaten_data[kabupaten_data['Jenis'] == jenis]
    
    years = np.array([int(year) for year in kabupaten_tahun]).reshape(-1, 1)
    production = jenis_data[kabupaten_tahun].values.flatten()
    
    model = LinearRegression()
    model.fit(years, production)
    
    tahun_prediksi_array = np.array(tahun_prediksi).reshape(-1, 1)
    future_predictions = model.predict(tahun_prediksi_array)
    predictions.append((jenis, tahun_prediksi, future_predictions))
    r2 = r2_score(production, future_predictions)
    mae = mean_absolute_error(production, future_predictions)
    plt.plot(years.flatten(), production, marker='o', label=f'{jenis} (Data)')
    plt.plot(tahun_prediksi, future_predictions, marker='x', linestyle='--', label=f'{jenis} (Prediksi)')

plt.title('Prediksi Produksi di Sulawesi Barat (2022-2026)', fontsize=14)
plt.xlabel('Tahun', fontsize=12)
plt.ylabel('Hasil Produksi', fontsize=12)
plt.legend(title='Jenis Produksi')
plt.grid(True)
plt.tight_layout()
plt.show()

# Menampilkan prediksi
for jenis, years, preds in predictions:
    print(f"Prediksi {jenis} untuk tahun 2022-2026:")
    for year, pred in zip(years, preds):
        print(f"Tahun {year}: {pred:.2f}")

print(f"Prediksi untuk {jenis} di Sulawesi Barat:")
print(f"R-squared: {r2:.2f}, MAE: {mae:.2f}")