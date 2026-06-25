import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
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
X = df.drop(columns=['stroke'])
y = df['stroke']

# Apply Wrapper Method using Recursive Feature Elimination (RFE) with Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(estimator=log_reg, n_features_to_select=1)  # Rank all features
rfe.fit(X, y)

# Get feature rankings
feature_rankings = pd.DataFrame({'Feature': X.columns, 'Ranking': rfe.ranking_})
feature_rankings.sort_values(by='Ranking', inplace=True)

# Display the feature ranking results
print("\nFeature Ranking using Wrapper Method (RFE with Logistic Regression):")
print(feature_rankings)

# Get the sorted features based on ranking (best features first)
sorted_features_wrapper = feature_rankings['Feature'].tolist()

# Initialize lists to store results for KNN
accuracies = []

# Repeat experiments 5 times
for experiment in range(5):
    # Shuffle the dataset randomly and split class-wise (70% training and 30% testing)
    df_shuffled = df.sample(frac=1, random_state=experiment).reset_index(drop=True)

    # Split the dataset class-wise into 70% training and 30% testing
    train_df = df_shuffled.groupby('stroke', group_keys=False).apply(lambda x: x.sample(frac=0.7, random_state=experiment))
    test_df = df_shuffled.drop(train_df.index)

    # Extract features and target for the current split
    X_train = train_df.drop(columns=['stroke'])
    y_train = train_df['stroke']
    X_test = test_df.drop(columns=['stroke'])
    y_test = test_df['stroke']

    # Initialize a list to store accuracies for each number of features
    accuracies_lda = []

    # Iteratively add features and evaluate KNN performance (K=4)
    for i in range(1, len(sorted_features_wrapper) + 1):
        selected_features = sorted_features_wrapper[:i]  # Select top i features

        # Train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=4)
        knn.fit(X_train[selected_features], y_train)
        y_pred = knn.predict(X_test[selected_features])

        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies_lda.append(accuracy)

    # Store the results for this experiment
    accuracies.append(accuracies_lda)

# Convert results to NumPy array for statistical analysis
accuracies = np.array(accuracies)

# Compute average and standard deviation for each number of features
average_accuracies = accuracies.mean(axis=0)
std_accuracies = accuracies.std(axis=0)

# Identify the best number of features based on highest average accuracy
best_num_features = np.argmax(average_accuracies) + 1  # +1 because we are using 1-based indexing for features
best_avg_accuracy = max(average_accuracies)
best_std_accuracy = std_accuracies[np.argmax(average_accuracies)]

# Display best number of features along with average accuracy and standard deviation
print(f"\nBest Number of Features for KNN using Wrapper Method (after 5 experiments): {best_num_features}")
print(f"Average Accuracy: {best_avg_accuracy:.4f}")
print(f"Standard Deviation: {best_std_accuracy:.4f}")

# Create a DataFrame to store results
knn_results = pd.DataFrame({'Number of Features': range(1, len(sorted_features_wrapper) + 1),
                            'Average Accuracy': average_accuracies, 'Std Dev': std_accuracies})

# Display results
print(f"\nKNN Accuracy for Different Number of Features (Wrapper Method, 5 Runs):")
print(knn_results)

# Plot results with annotation for best number of features
plt.figure(figsize=(8, 5))
plt.errorbar(range(1, len(sorted_features_wrapper) + 1), average_accuracies, yerr=std_accuracies, fmt='-o', label="Accuracy ± Std Dev")
plt.xlabel("Number of Features Used")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs Number of Features (Wrapper Method, 5 Runs)")
plt.xticks(range(1, len(sorted_features_wrapper) + 1))
plt.grid()

# Annotate the best number of features
plt.scatter(best_num_features, best_avg_accuracy, color='red', s=100, label=f"Best Features={best_num_features}, Accuracy={best_avg_accuracy:.4f}")
plt.legend()
plt.show()

# Get the highly discriminating feature (top-ranked)
best_feature_wrapper = sorted_features_wrapper[0]

# Initialize lists to store results for KNN with the most discriminating feature
k_values = list(range(1, 21))
accuracies_k = []

# Repeat experiments 5 times for the best discriminating feature
for experiment in range(5):
    # Shuffle the dataset randomly and split class-wise (70% training and 30% testing)
    df_shuffled = df.sample(frac=1, random_state=experiment).reset_index(drop=True)

    # Split the dataset class-wise into 70% training and 30% testing
    train_df = df_shuffled.groupby('stroke', group_keys=False).apply(lambda x: x.sample(frac=0.7, random_state=experiment))
    test_df = df_shuffled.drop(train_df.index)

    # Extract features and target for the best discriminating feature
    X_train = train_df[[best_feature_wrapper]]
    y_train = train_df['stroke']
    X_test = test_df[[best_feature_wrapper]]
    y_test = test_df['stroke']

    # Store accuracy for different K values
    accuracies_k_run = []

    # Iterate over different values of K (1 to 20) using only the best feature
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies_k_run.append(accuracy)

    # Store results for this experiment
    accuracies_k.append(accuracies_k_run)

# Convert results to NumPy array for statistical analysis
accuracies_k = np.array(accuracies_k)

# Compute average and standard deviation for each K value
average_accuracies_k = accuracies_k.mean(axis=0)
std_accuracies_k = accuracies_k.std(axis=0)

# Identify the best K value based on highest average accuracy
best_k = k_values[np.argmax(average_accuracies_k)]
best_accuracy = max(average_accuracies_k)

# Display best K value
print(f"\nBest K Value for Feature '{best_feature_wrapper}': {best_k} with Accuracy: {best_accuracy:.4f}")

# Create a DataFrame to store results
knn_results_k = pd.DataFrame({'K': k_values, 'Average Accuracy': average_accuracies_k, 'Std Dev': std_accuracies_k})

# Display results
print("\nKNN Accuracy for Different K Values (Best Feature):")
print(knn_results_k)

# Plot results with annotation for best K value
plt.figure(figsize=(8, 5))
plt.errorbar(k_values, average_accuracies_k, yerr=std_accuracies_k, fmt='-o', label="Accuracy ± Std Dev")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title(f"KNN Accuracy vs K for Feature: {best_feature_wrapper}")
plt.xticks(k_values)
plt.grid()

# Annotate the best K value
plt.scatter(best_k, best_accuracy, color='red', s=100, label=f"Best K={best_k}, Accuracy={best_accuracy:.4f}")
plt.legend()
plt.show()
