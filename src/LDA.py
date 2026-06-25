import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

# Initialize lists to store experiment results
k_values = list(range(1, 21))
experiment_results = []

# Repeat experiments 5 times
for experiment in range(5):
    # Shuffle the dataset randomly
    df = df.sample(frac=1, random_state=experiment).reset_index(drop=True)

    # Split the dataset class-wise into 70% training and 30% testing
    train_df = df.groupby('stroke', group_keys=False).apply(lambda x: x.sample(frac=0.7, random_state=experiment))
    test_df = df.drop(train_df.index)

    # Extract features and target
    X_train = train_df.drop(columns=['stroke'])
    y_train = train_df['stroke']
    X_test = test_df.drop(columns=['stroke'])
    y_test = test_df['stroke']

    # Apply LDA (Linear Discriminant Analysis)
    lda = LinearDiscriminantAnalysis()
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    # Store accuracy for different K values
    accuracies_lda = []

    # Iterate over different values of K (1 to 20) using LDA-transformed data
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_lda, y_train)
        y_pred_lda = knn.predict(X_test_lda)

        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred_lda)
        accuracies_lda.append(accuracy)

    # Store results for this experiment
    experiment_results.append(accuracies_lda)

# Convert results to NumPy array for statistical analysis
experiment_results = np.array(experiment_results)

# Compute average and standard deviation for each K value
average_accuracies = experiment_results.mean(axis=0)
std_accuracies = experiment_results.std(axis=0)

# Identify the best K value based on highest average accuracy
best_k_lda = k_values[np.argmax(average_accuracies)]
best_avg_accuracy_lda = max(average_accuracies)
best_std_accuracy_lda = std_accuracies[np.argmax(average_accuracies)]

# Display best K value along with average accuracy and standard deviation
print(f"\nBest K Value for KNN using LDA-transformed data (after 5 experiments): {best_k_lda}")
print(f"Average Accuracy: {best_avg_accuracy_lda:.4f}")
print(f"Standard Deviation: {best_std_accuracy_lda:.4f}")

# Create a DataFrame to store results
knn_results_lda = pd.DataFrame({'K': k_values, 'Average Accuracy': average_accuracies, 'Std Dev': std_accuracies})

# Display results
print("\nKNN Accuracy for Different K Values (LDA with 5 Runs):")
print(knn_results_lda)

# Plot results with annotation for best K value
plt.figure(figsize=(8, 5))
plt.errorbar(k_values, average_accuracies, yerr=std_accuracies, fmt='-o', label="Accuracy ± Std Dev")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K (LDA Transformed Data, 5 Runs)")
plt.xticks(k_values)
plt.grid()

# Annotate the best K value
plt.scatter(best_k_lda, best_avg_accuracy_lda, color='red', s=100, label=f"Best K={best_k_lda}, Accuracy={best_avg_accuracy_lda:.4f}")
plt.legend()
plt.show()
