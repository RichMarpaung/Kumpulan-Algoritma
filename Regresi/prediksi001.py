import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Data Majene
file_path = 'PRoduksi.csv'  # Sesuaikan dengan path file Anda
data = pd.read_csv(file_path)

majene_data = data[data['Kabupaten dan Provinsi'].str.contains('Majene')]
years = [2015, 2016, 2017, 2018, 2020, 2021]
production = majene_data.iloc[0][years].values

# Siapkan data untuk pelatihan
X_train = np.array(years).reshape(-1, 1)  # Tahun sebagai fitur
y_train = production  # Produksi sebagai target

# Latih model Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Ekstrapolasi untuk 2022–2031
future_years = np.array(range(2022, 2032)).reshape(-1, 1)
future_predictions = rf_model.predict(future_years)

# Hasil prediksi
future_df = pd.DataFrame({
    "Year": range(2022, 2032),
    "Predicted Production": future_predictions
})

print(future_df)
