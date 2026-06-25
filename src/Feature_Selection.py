import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

# Load the dataset with robust path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "../data/healthcare-dataset-stroke-data.csv")
df = pd.read_csv(file_path)

# Drop the ID column as it's not a relevant feature
df.drop(columns=['id'], inplace=True)

# Encode categorical variables
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Handle missing values by filling with median
df.fillna(df.median(), inplace=True)

# Split features and target
X = df.drop(columns=['stroke'])
y = df['stroke']

# Select the specific features for KNN
selected_features = ['age', 'avg_glucose_level', 'heart_disease']

# Initialize lists to store results
k_values = list(range(1, 21))
accuracies = []
experiment_results = []

# Repeat experiments 5 times
for experiment in range(5):
    # Shuffle the dataset randomly and split class-wise
    df_shuffled = df.sample(frac=1, random_state=experiment).reset_index(drop=True)

    # Split the dataset class-wise into 70% training and 30% testing
    train_df = df_shuffled.groupby('stroke', group_keys=False).apply(lambda x: x.sample(frac=0.7, random_state=experiment))
    test_df = df_shuffled.drop(train_df.index)

    # Extract features and target for training and testing
    X_train = train_df[selected_features]
    y_train = train_df['stroke']
    X_test = test_df[selected_features]
    y_test = test_df['stroke']

    # Initialize a list to store accuracies for different values of K
    accuracies_experiment = []

    # Iterate over different values of K (1 to 20) using selected features
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies_experiment.append(accuracy)

    # Store results for this experiment
    experiment_results.append(accuracies_experiment)

# Convert results to NumPy array for statistical analysis
experiment_results = np.array(experiment_results)

# Compute average and standard deviation for each K value
average_accuracies = experiment_results.mean(axis=0)
std_accuracies = experiment_results.std(axis=0)

# Identify the best K value based on highest average accuracy
best_k = k_values[np.argmax(average_accuracies)]
best_accuracy = max(average_accuracies)

# Display best K value
print(f"\nBest K Value for Features {selected_features}: {best_k} with Average Accuracy: {best_accuracy:.4f}")
print(f"Standard Deviation: {std_accuracies[np.argmax(average_accuracies)]:.4f}")

# Create a DataFrame to store results
knn_results = pd.DataFrame({'K': k_values, 'Average Accuracy': average_accuracies, 'Std Dev': std_accuracies})

# Display results
print("\nKNN Accuracy for Different K Values:")
print(knn_results)

# Plot results with annotation for best K value
plt.figure(figsize=(8, 5))
plt.errorbar(k_values, average_accuracies, yerr=std_accuracies, fmt='-o', label="Accuracy ± Std Dev")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title(f"KNN Accuracy vs K for Features: {', '.join(selected_features)}")
plt.xticks(k_values)
plt.grid()

# Annotate the best K value
plt.scatter(best_k, best_accuracy, color='red', s=100, label=f"Best K={best_k}, Accuracy={best_accuracy:.4f}")
plt.legend()
plt.show()
