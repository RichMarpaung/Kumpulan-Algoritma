import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

file_path = 'Produksi001.csv'  
data = pd.read_csv(file_path)

kabupaten_list = data['Kabupaten'].unique()
future_years = [2022, 2023, 2024,2025, 2026, 2027, 2028, 2030]

all_predictions = []

n_kabupaten = len(kabupaten_list)
fig, axes = plt.subplots(nrows=(n_kabupaten + 1) // 2, ncols=2, figsize=(15, 5 * ((n_kabupaten + 1) // 2)))

axes = axes.flatten()

for idx, kabupaten in enumerate(kabupaten_list):
    ax = axes[idx]  # Akses subplot sesuai indeks
    kabupaten_data = data[data['Kabupaten'] == kabupaten]
    jenis_list = kabupaten_data['Jenis'].unique()
    
    for jenis in jenis_list:
        jenis_data = kabupaten_data[kabupaten_data['Jenis'] == jenis]
        
        available_years = [col for col in jenis_data.columns if col.isdigit()]
        
        years = np.array([int(year) for year in available_years]).reshape(-1, 1)
        production = jenis_data[available_years].values.flatten()
        
        model = LinearRegression()
        model.fit(years, production)
        
        predictions_on_training = model.predict(years)
        r2 = r2_score(production, predictions_on_training)
        mae = mean_absolute_error(production, predictions_on_training)
        
        future_years_array = np.array(future_years).reshape(-1, 1)
        future_predictions = model.predict(future_years_array)
        all_predictions.append((kabupaten, jenis, future_years, future_predictions, r2, mae))
        
    ax.plot(years.flatten(), production, marker='o', label=f'{jenis} (Data)')
    ax.plot(future_years, future_predictions, marker='x', linestyle='--', label=f'{jenis} (Prediksi)')
    
    ax.set_title(f'Prediksi Produksi di {kabupaten}', fontsize=14)
    ax.set_xlabel('Tahun', fontsize=12)
    ax.set_ylabel('Hasil Produksi', fontsize=12)
    ax.legend(title='Jenis Produksi')
    ax.grid(True)

for i in range(n_kabupaten, len(axes)):
    fig.delaxes(axes[i])
    
for kabupaten, jenis, years, preds, r2, mae in all_predictions:
    print(f"Prediksi untuk {jenis} di {kabupaten}:")
    print(f"R-squared: {r2:.2f}, MAE: {mae:.2f}")
    for year, pred in zip(years, preds):
        print(f"Tahun {year}: {pred:.2f}")
plt.tight_layout()
plt.show()


