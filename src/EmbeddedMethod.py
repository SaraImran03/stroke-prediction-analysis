import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
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
X = df.drop(columns=['stroke'])  # Only features
y = df['stroke']  # Target variable

# Apply Embedded Method using RandomForestClassifier to get feature importances
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

# Display feature importance results
print("\nFeature Importance (Embedded Method - Random Forest):")
print(feature_importances)

# Get the sorted features based on importance
sorted_features = feature_importances['Feature'].tolist()

# Initialize lists to store results
feature_subset = []
accuracies = []
experiment_results = []

# Repeat experiments 5 times
for experiment in range(5):
    # Shuffle the dataset randomly and split class-wise
    df_shuffled = df.sample(frac=1, random_state=experiment).reset_index(drop=True)

    # Split the dataset class-wise into 70% training and 30% testing
    train_df = df_shuffled.groupby('stroke', group_keys=False).apply(lambda x: x.sample(frac=0.7, random_state=experiment))
    test_df = df_shuffled.drop(train_df.index)

    # Extract features and target
    X_train = train_df.drop(columns=['stroke'])
    y_train = train_df['stroke']
    X_test = test_df.drop(columns=['stroke'])
    y_test = test_df['stroke']

    # Iteratively add features and evaluate KNN performance
    accuracies_lda = []
    for i in range(1, len(sorted_features) + 1):
        selected_features = sorted_features[:i]  # Select top i features
        feature_subset.append(", ".join(selected_features))  # Store feature names

        # Train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=4)
        knn.fit(X_train[selected_features], y_train)
        y_pred = knn.predict(X_test[selected_features])

        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies_lda.append(accuracy)

    experiment_results.append(accuracies_lda)

# Convert results to NumPy array for statistical analysis
experiment_results = np.array(experiment_results)

# Compute average and standard deviation for each number of features
average_accuracies = experiment_results.mean(axis=0)
std_accuracies = experiment_results.std(axis=0)

# Identify the best number of features based on highest average accuracy
best_num_features = np.argmax(average_accuracies) + 1  # +1 because we are using 1-based indexing for features
best_avg_accuracy = max(average_accuracies)
best_std_accuracy = std_accuracies[np.argmax(average_accuracies)]

# Display best number of features along with average accuracy and standard deviation
print(f"\nBest Number of Features for KNN using Embedded Method (after 5 experiments): {best_num_features}")
print(f"Average Accuracy: {best_avg_accuracy:.4f}")
print(f"Standard Deviation: {best_std_accuracy:.4f}")

# Create a DataFrame to store results
knn_results = pd.DataFrame({'Number of Features': range(1, len(sorted_features) + 1),
                            'Average Accuracy': average_accuracies, 'Std Dev': std_accuracies})

# Display results
print(f"\nKNN Accuracy for Different K Values (Embedded Method, 5 Runs):")
print(knn_results)

# Plot results with annotation for best number of features
plt.figure(figsize=(8, 5))
plt.errorbar(range(1, len(sorted_features) + 1), average_accuracies, yerr=std_accuracies, fmt='-o', label="Accuracy ± Std Dev")
plt.xlabel("Number of Features Used")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs Number of Features (Embedded Method, 5 Runs)")
plt.xticks(range(1, len(sorted_features) + 1))
plt.grid()

# Annotate the best number of features
plt.scatter(best_num_features, best_avg_accuracy, color='red', s=100, label=f"Best Features={best_num_features}, Accuracy={best_avg_accuracy:.4f}")
plt.legend()
plt.show()

# Get the highly discriminating feature (top-ranked)
best_feature = sorted_features[0]

# Initialize lists to store results for KNN with the most discriminating feature
k_values = list(range(1, 21))
experiment_results_best_feature = []

# Repeat experiments 5 times for the best discriminating feature
for experiment in range(5):
    # Shuffle the dataset randomly and split class-wise
    df_shuffled = df.sample(frac=1, random_state=experiment).reset_index(drop=True)

    # Split the dataset class-wise into 70% training and 30% testing
    train_df = df_shuffled.groupby('stroke', group_keys=False).apply(lambda x: x.sample(frac=0.7, random_state=experiment))
    test_df = df_shuffled.drop(train_df.index)

    # Extract features and target for the best discriminating feature
    X_train = train_df[[best_feature]]
    y_train = train_df['stroke']
    X_test = test_df[[best_feature]]
    y_test = test_df['stroke']

    # Store accuracy for different K values
    accuracies_best_feature = []

    # Iterate over different values of K (1 to 20) using only the best feature
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies_best_feature.append(accuracy)

    # Store results for this experiment
    experiment_results_best_feature.append(accuracies_best_feature)

# Convert results to NumPy array for statistical analysis
experiment_results_best_feature = np.array(experiment_results_best_feature)

# Compute average and standard deviation for each K value
average_accuracies_best_feature = experiment_results_best_feature.mean(axis=0)
std_accuracies_best_feature = experiment_results_best_feature.std(axis=0)

# Identify the best K value based on highest average accuracy
best_k_best_feature = k_values[np.argmax(average_accuracies_best_feature)]
best_avg_accuracy_best_feature = max(average_accuracies_best_feature)
best_std_accuracy_best_feature = std_accuracies_best_feature[np.argmax(average_accuracies_best_feature)]

# Display best K value along with average accuracy and standard deviation
print(f"\nBest K Value for KNN using best feature ({best_feature}) (after 5 experiments): {best_k_best_feature}")
print(f"Average Accuracy: {best_avg_accuracy_best_feature:.4f}")
print(f"Standard Deviation: {best_std_accuracy_best_feature:.4f}")

# Create a DataFrame to store results
knn_results_best_feature = pd.DataFrame({'K': k_values, 'Average Accuracy': average_accuracies_best_feature, 'Std Dev': std_accuracies_best_feature})

# Display results
print(f"\nKNN Accuracy for Different K Values (Best Feature {best_feature}):")
print(knn_results_best_feature)

# Plot results with annotation for best K value
plt.figure(figsize=(8, 5))
plt.errorbar(k_values, average_accuracies_best_feature, yerr=std_accuracies_best_feature, fmt='-o', label="Accuracy ± Std Dev")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title(f"KNN Accuracy vs K (Best Feature: {best_feature}, 5 Runs)")
plt.xticks(k_values)
plt.grid()

# Annotate the best K value
plt.scatter(best_k_best_feature, best_avg_accuracy_best_feature, color='red', s=100, label=f"Best K={best_k_best_feature}, Accuracy={best_avg_accuracy_best_feature:.4f}")
plt.legend()
plt.show()
