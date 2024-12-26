import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Produksi001.csv'
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

# Prepare predictions
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
    
    # Plot actual and predicted values
    plt.plot(years.flatten(), production, marker='o', label=f'{jenis} (Data)')
    plt.scatter(future_years, [8000 if p == "Rendah" else 8250 if p == "Sedang" else 8750 for p in future_predictions], 
                marker='x', label=f'{jenis} (Prediksi)')

# Plot configuration
plt.title('Prediksi Kategori Produksi di Majene dengan Naive Bayes (2025-2030)', fontsize=14)
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
