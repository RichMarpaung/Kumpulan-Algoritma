import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file_path = 'data/water_potability.csv'
data = pd.read_csv(file_path)

data = data.fillna(data.median())

X = data.drop('Potability', axis=1)
y = data['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def calculate_prior(y):
    classes = np.unique(y)
    priors = {c: np.mean(y == c) for c in classes}
    return priors

def calculate_statistics(X, y):
    statistics = {}
    for c in np.unique(y):
        X_c = X[y == c]
        statistics[c] = {
            "mean": X_c.mean(axis=0),
            "std": X_c.std(axis=0)
        }
    return statistics

def gaussian_probability(x, mean, std):
    exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

def calculate_posteriors(X, priors, statistics):
    posteriors = []
    for x in X:
        class_probs = {}
        for c, prior in priors.items():
            class_probs[c] = np.log(prior)
            for i in range(len(x)):
                mean = statistics[c]["mean"][i]
                std = statistics[c]["std"][i]
                class_probs[c] += np.log(gaussian_probability(x[i], mean, std))
        posteriors.append(class_probs)
    return posteriors

def predict(X, priors, statistics):
    posteriors = calculate_posteriors(X, priors, statistics)
    predictions = [max(p, key=p.get) for p in posteriors]
    return predictions

priors = calculate_prior(y_train)
statistics = calculate_statistics(X_train.values, y_train.values)

y_pred_manual = predict(X_test.values, priors, statistics)

manual_accuracy = accuracy_score(y_test, y_pred_manual)
print("Manual Naive Bayes Accuracy:", manual_accuracy)
