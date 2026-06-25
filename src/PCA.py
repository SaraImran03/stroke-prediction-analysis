import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

# Standardize the dataset (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to apply PCA with a given number of components and run KNN
def run_knn_with_pca(num_components):
    # Apply PCA with the specified number of components
    pca_model = PCA(n_components=num_components)
    X_pca_selected = pca_model.fit_transform(X_scaled)

    # Initialize lists to store results
    k_values = list(range(1, 21))
    accuracies_pca = []
    experiment_results = []

    # Repeat experiments 5 times
    for experiment in range(5):
        # Shuffle the dataset randomly
        df_shuffled = df.sample(frac=1, random_state=experiment).reset_index(drop=True)

        # Split the dataset class-wise into 70% training and 30% testing
        train_df = df_shuffled.groupby('stroke', group_keys=False).apply(lambda x: x.sample(frac=0.7, random_state=experiment))
        test_df = df_shuffled.drop(train_df.index)

        # Extract features and target
        X_train = train_df.drop(columns=['stroke'])
        y_train = train_df['stroke']
        X_test = test_df.drop(columns=['stroke'])
        y_test = test_df['stroke']

        # Apply PCA (Linear Discriminant Analysis)
        pca_model = PCA(n_components=num_components)
        X_train_lda = pca_model.fit_transform(X_train, y_train)
        X_test_lda = pca_model.transform(X_test)

        # Store accuracy for different K values
        accuracies_lda = []

        # Iterate over different values of K (1 to 20) using PCA-transformed data
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
    best_k_pca = k_values[np.argmax(average_accuracies)]
    best_avg_accuracy_pca = max(average_accuracies)
    best_std_accuracy_pca = std_accuracies[np.argmax(average_accuracies)]

    # Display best K value along with average accuracy and standard deviation
    print(f"\nBest K Value for KNN using PCA ({num_components} Components): {best_k_pca}")
    print(f"Average Accuracy: {best_avg_accuracy_pca:.4f}")
    print(f"Standard Deviation: {best_std_accuracy_pca:.4f}")

    # Create a DataFrame to store results
    knn_results_pca = pd.DataFrame({'K': k_values, 'Average Accuracy': average_accuracies, 'Std Dev': std_accuracies})

    # Display results
    print(f"\nKNN Accuracy for Different K Values (PCA with {num_components} Components):")
    print(knn_results_pca)

    # Plot results with annotation for best K value
    plt.figure(figsize=(8, 5))
    plt.errorbar(k_values, average_accuracies, yerr=std_accuracies, fmt='-o', label="Accuracy ± Std Dev")
    plt.xlabel("K Value")
    plt.ylabel("Accuracy")
    plt.title(f"KNN Accuracy vs K (PCA with {num_components} Components, 5 Runs)")
    plt.xticks(k_values)
    plt.grid()

    # Annotate the best K value
    plt.scatter(best_k_pca, best_avg_accuracy_pca, color='red', s=100, label=f"Best K={best_k_pca}, Accuracy={best_avg_accuracy_pca:.4f}")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Run KNN with PCA for 8 components
    run_knn_with_pca(8)

    # Run KNN with PCA for 9 components
    run_knn_with_pca(9)
