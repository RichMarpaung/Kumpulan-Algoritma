import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
file_path = 'PRoduksi.csv'  # Sesuaikan dengan path file Anda
data = pd.read_csv(file_path)

# Filter data untuk jenis "Kakao"
kakao_data = data[data['Jenis'].str.contains('Kakao', case=False)]

# Ambil tahun sebagai fitur
years = ['2015', '2016', '2017', '2018', '2020', '2021']
X = kakao_data[years].astype(float)  # Pastikan kolom tahun bertipe numerik

# Latih model untuk setiap daerah
predictions = {}
future_years = np.array(range(2022, 2032)).reshape(-1, 1)  # Tahun 2022-2031

for _, row in kakao_data.iterrows():
    region = row['Kabupaten dan Provinsi']
    y = row[years].values
    
    # Siapkan model
    model = RandomForestRegressor(random_state=42)
    model.fit(np.array(years).reshape(-1, 1), y)
    
    # Prediksi produksi untuk 2022-2031
    future_production = model.predict(future_years)
    predictions[region] = future_production

# Hasil prediksi
predicted_df = pd.DataFrame(predictions, index=range(2022, 2032))
print(predicted_df)
import matplotlib.pyplot as plt

# Visualisasi prediksi untuk tiap daerah
plt.figure(figsize=(12, 8))
for region in predicted_df.columns:
    plt.plot(predicted_df.index, predicted_df[region], label=region)

# Atur grafik
plt.title("Prediksi Produksi Kakao (2022–2031)", fontsize=16)
plt.xlabel("Tahun", fontsize=12)
plt.ylabel("Produksi (Ton)", fontsize=12)
plt.legend(title="Daerah", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Tampilkan grafik
plt.show()

