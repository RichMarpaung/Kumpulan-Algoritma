import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
file_path = 'Produksi002.csv'  # Sesuaikan nama file
data = pd.read_csv(file_path)

majene_data = data[data['Kabupaten'].str.contains("Majene", case=False)]

majene_years = ['2015', '2016', '2018', '2020', '2021']
majene_types = majene_data['Jenis'].unique()

future_years = [2022,2023,2024,2025, 2026,2027,2028,2030]
predictions = []

for jenis in majene_types:
    jenis_data = majene_data[majene_data['Jenis'] == jenis]
    
    years = np.array([int(year) for year in majene_years]).reshape(-1, 1)
    production = jenis_data[majene_years].values.flatten()
    
    model = LinearRegression()
    model.fit(years, production)
    
    future_years_array = np.array(future_years).reshape(-1, 1)
    future_predictions = model.predict(future_years_array)
    predictions.append((jenis, future_years, future_predictions))
    
    plt.plot(years.flatten(), production, marker='o', label=f'{jenis} (Data)')
    plt.plot(future_years, future_predictions, marker='x', linestyle='--', label=f'{jenis} (Prediksi)')

plt.title('Prediksi Produksi di Majene (2022-2026)', fontsize=14)
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
