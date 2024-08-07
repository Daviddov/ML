import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Ex7.csv")

# Describe the dataset
stat = df.describe()
print("Dataset Statistics:")
print(stat)

# Prepare features (X) and target (y)
X = df.drop(["Purchased", "User ID", "Gender"], axis=1)
y = df["Purchased"]

# Standardize the features
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predict probabilities for the test set
y_pred = model.predict(X_test)
prob = model.predict_proba(X_test)

accuracy= metrics.accuracy_score(y_test, y_pred)

# Print model evaluation metrics
print("\nModel Evaluation:")
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {metrics.precision_score(y_test, y_pred):.4f}")
print(f"Recall: {metrics.recall_score(y_test, y_pred):.4f}")
print(f"F1-score: {metrics.f1_score(y_test, y_pred):.4f}")


arr = []
ages = range(1, 121)  # Ages from 1 to 120
probabilities = []

for age in ages:
    # Prepare new data for prediction
    new_data = np.array([[age, 24500]])
    scaled_new_data = sc.transform(new_data)
    pred_prob = model.predict_proba(scaled_new_data)
    probabilities.append(pred_prob[0][1])  # Probability of positive class
    if pred_prob[0][1] > 0.7:
        arr.append(age)

# Prepare new data for prediction
new_data = np.array([[99, 330000]])
scaled_new_data = sc.transform(new_data)
pred = model.predict(scaled_new_data)
pred_prob = model.predict_proba(scaled_new_data)

print(f"\nPrediction for the new data: {pred[0]}")
print(f"Probability of purchase: {pred_prob[0][1]:.4f}")

