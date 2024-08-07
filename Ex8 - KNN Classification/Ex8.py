import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def find_best_K(from_number, to_number):
    # Find the best K
    k_range = range(from_number, to_number)
    accuracies = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"K = {k}, Accuracy: {accuracy:.4f}")

    # Find the best K
    best_accuracy = max(accuracies)
    best_k = k_range[accuracies.index(best_accuracy)]
    
    print(f"\nBest K: {best_k}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    # Plot K vs Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, accuracies, marker='o')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.title('K-Value vs Accuracy for KNN Classifier')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()
    return best_k

# Load the dataset
df = pd.read_csv("Ex8.csv")

# Prepare features (X) and target (y)
X = df.drop(["Purchased", "User ID", "Gender"], axis=1)
y = df["Purchased"]

# Standardize the features
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


best_k = find_best_K(2, 10)

# Train the model with the best K
best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate and print the final accuracy
final_accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Model Accuracy: {final_accuracy:.4f}")

# Prepare new data for prediction
new_data = np.array([[33, 150000]])
scaled_new_data = sc.transform(new_data)
pred = best_model.predict(scaled_new_data)
pred_prob = best_model.predict_proba(scaled_new_data)

print(f"\nPrediction for the new data: {pred[0]}")
print(f"Probability of purchase: {pred_prob[0][1]:.4f}")