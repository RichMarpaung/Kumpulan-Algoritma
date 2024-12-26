import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'data/water_potability.csv'
data = pd.read_csv(file_path)

# Step 1: Handle missing values (fill with the median value of each column)
data = data.fillna(data.median())

# Step 2: Split the dataset into features (X) and target (y)
X = data.drop('Potability', axis=1)
y = data['Potability']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the Naive Bayes classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Step 5: Make predictions and evaluate the model
y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy of Naive Bayes model:", accuracy)
