import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os

def knn_experiment(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    df.drop(columns=['id'], inplace=True)  # Drop 'id' column

    # Handle missing values
    df['bmi'].fillna(df['bmi'].median(), inplace=True)

    # Encode categorical variables
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    df_encoded = df.copy()
    encoder = LabelEncoder()

    for col in categorical_cols:
        df_encoded[col] = encoder.fit_transform(df_encoded[col])

    # Split dataset
    X = df_encoded.drop(columns=['stroke'])
    y = df_encoded['stroke']

    # Define K values to test
    k_values = range(1, 21)
    accuracies = []

    # Repeat experiment 5 times
    for experiment in range(5):
        # Shuffle the dataset randomly
        df_shuffled = df_encoded.sample(frac=1, random_state=experiment).reset_index(drop=True)

        # Split the dataset class-wise into 70% training and 30% testing
        train_df = df_shuffled.groupby('stroke', group_keys=False).apply(lambda x: x.sample(frac=0.7, random_state=experiment))
        test_df = df_shuffled.drop(train_df.index)

        # Extract features and target for the current split
        X_train = train_df.drop(columns=['stroke'])
        y_train = train_df['stroke']
        X_test = test_df.drop(columns=['stroke'])
        y_test = test_df['stroke']

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Store accuracies for each K value
        experiment_accuracies = []

        # Experiment with different k values
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            experiment_accuracies.append(accuracy)

        # Append accuracies for this experiment
        accuracies.append(experiment_accuracies)

    # Convert results to NumPy array for statistical analysis
    accuracies = np.array(accuracies)

    # Compute average and standard deviation for each k
    average_accuracies = accuracies.mean(axis=0)
    std_accuracies = accuracies.std(axis=0)

    # Identify the best k value based on highest average accuracy
    best_k = k_values[np.argmax(average_accuracies)]
    best_accuracy = max(average_accuracies)

    # Display results
    print(f"Best k value: {best_k} with Average Accuracy: {best_accuracy:.2f}")
    print(f"Standard Deviation: {std_accuracies[np.argmax(average_accuracies)]:.2f}")

    # Create a DataFrame to store results
    knn_results = pd.DataFrame({'K': k_values, 'Average Accuracy': average_accuracies, 'Std Dev': std_accuracies})

    # Display results
    print("\nKNN Accuracy for Different K Values (5 Runs):")
    print(knn_results)

    # Plot results with error bars for standard deviation
    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, average_accuracies, yerr=std_accuracies, fmt='-o', label="Accuracy ± Std Dev")
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('KNN Accuracy vs. K Value (5 Runs)')
    plt.xticks(k_values)
    plt.grid()

    # Annotate the best k value
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k = {best_k} (Accuracy: {best_accuracy:.2f})')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example usage with robust relative path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "../data/healthcare-dataset-stroke-data.csv")
    knn_experiment(file_path)
