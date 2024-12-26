import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Produksi001.csv'  # Ubah sesuai dengan lokasi file Anda
data = pd.read_csv(file_path)

# Filter data for "Majene"
majene_data = data[data['Kabupaten'].str.contains("Majene", case=False)]

# Define the years and types
majene_years = ['2015', '2016', '2018', '2020', '2021']
majene_types = majene_data['Jenis'].unique()

# Define future years for prediction
future_years = [2025, 2026, 2027, 2028, 2030]

# Function to categorize production values
def categorize_production(values):
    categories = []
    for value in values:
        if value < 8000:
            categories.append("Rendah")
        elif 8000 <= value < 8500:
            categories.append("Sedang")
        else:
            categories.append("Tinggi")
    return np.array(categories)

# Prepare predictions and categories
predictions = []

for jenis in majene_types:
    # Filter data for the specific type
    jenis_data = majene_data[majene_data['Jenis'] == jenis]
    
    # Prepare data for training
    years = np.array([int(year) for year in majene_years]).reshape(-1, 1)
    production = jenis_data[majene_years].values.flatten()
    production_categories = categorize_production(production)
    
    # Train Gaussian Naive Bayes model
    model = GaussianNB()
    model.fit(years, production_categories)
    
    # Predict for future years
    future_years_array = np.array(future_years).reshape(-1, 1)
    future_predictions = model.predict(future_years_array)
    predictions.append((jenis, future_years, future_predictions))
    
    # Map categories back for visualization
    category_map = {"Rendah": 7000, "Sedang": 8250, "Tinggi": 8750}
    future_predictions_mapped = [category_map[cat] for cat in future_predictions]
    
    # Plot actual and predicted values
    plt.plot(years.flatten(), production, marker='o', label=f'{jenis} (Data)')
    plt.plot(future_years, future_predictions_mapped, marker='x', linestyle='--', label=f'{jenis} (Prediksi)')

# Plot configuration
plt.title('Prediksi Kategori Produksi di Majene (2025-2030)', fontsize=14)
plt.xlabel('Tahun', fontsize=12)
plt.ylabel('Kategori Produksi (Rendah/Sedang/Tinggi)', fontsize=12)
plt.legend(title='Jenis Produksi')
plt.grid(True)
plt.tight_layout()
plt.show()

# Display predictions
for jenis, years, preds in predictions:
    print(f"Prediksi kategori {jenis} untuk tahun 2025-2030:")
    for year, pred in zip(years, preds):
        print(f"Tahun {year}: {pred}")
