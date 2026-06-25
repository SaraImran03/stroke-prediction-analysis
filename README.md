# Stroke Risk Prediction: Feature Engineering, Selection, and Dimensionality Reduction Analysis

An analytical machine learning project evaluating the impact of various feature selection and dimensionality reduction techniques on the classification performance of a $k$-Nearest Neighbors ($k\text{NN}$) classifier. 

Using the **Healthcare Stroke Dataset**, this project models stroke risk and compares baseline performance against Filter, Wrapper, and Embedded feature selection methods, as well as Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA).

---

## Project Overview

Stroke is a leading cause of death and disability globally. Early prediction of stroke risk using clinical and demographic features can significantly improve patient outcomes. This project investigates how to build a highly accurate yet computationally efficient classification model by reducing data dimensionality while maintaining robust prediction rates.

### Core Objectives
1. **Baseline Modeling**: Implement a $k\text{NN}$ classifier using all raw features, optimizing the hyperparameter $k$.
2. **Feature Selection**: Apply and compare:
   - **Filter Method**: ANOVA $F$-value ranking (`SelectKBest`).
   - **Wrapper Method**: Recursive Feature Elimination (RFE) with Logistic Regression.
   - **Embedded Method**: Random Forest feature importance.
3. **Dimensionality Reduction**: Apply and compare:
   - **Principal Component Analysis (PCA)**: Unsupervised linear projection.
   - **Linear Discriminant Analysis (LDA)**: Supervised linear projection.
4. **Statistical Validation**: Use class-wise stratified 70:30 train-test splits repeated over 5 runs to calculate average accuracy and standard deviation.

---

## Dataset Description

The dataset contains clinical and lifestyle factors for patients, with the target variable `stroke` (1 if the patient had a stroke, 0 otherwise).

### Features:
- **Numerical**: `age`, `avg_glucose_level`, `bmi`
- **Categorical**: `gender`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`

*Note: Missing values in `bmi` are filled using the median value, and categorical features are encoded using label encoding.*

---

## Methodology & Implementations

### 1. Baseline Model ($k\text{NN}$)
A baseline $k\text{NN}$ classifier was trained on all 10 features. The model was evaluated for $k \in [1, 20]$ across 5 randomized runs.
- **Best Hyperparameter**: $k = 8$
- **Average Accuracy**: **95.12%**
- **Standard Deviation**: **0.00026**

### 2. Feature Selection Methods
To identify the most predictive features, three selection approaches were analyzed:

- **Filter Method (ANOVA $F$-test)**:
  - Ranked the features based on variance analysis. The top-ranked feature was `age` (Score: 326.92), followed by `heart_disease` (94.70) and `avg_glucose_level` (90.50).
  - Iteratively adding features showed that using **only the single top feature (`age`)** yielded optimal accuracy (**95.11%** at $k=1$).

- **Embedded Method (Random Forest Importance)**:
  - Feature ranking: `avg_glucose_level` (0.286), `bmi` (0.235), `age` (0.231).
  - Evaluating performance showed that using **only `avg_glucose_level`** achieved **95.11%** accuracy at $k=6$.

- **Wrapper Method (RFE with Logistic Regression)**:
  - Feature ranking: `heart_disease` (Rank 1), `ever_married` (Rank 2), `hypertension` (Rank 3).
  - Using **only `heart_disease`** achieved **95.11%** accuracy at $k=1$.

#### Feature Selection Comparison Table

| Selection Method | Best Feature | Optimal $k$ | No. of Features | Average Accuracy | Standard Deviation |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Filter (ANOVA)** | `age` | 1 | 1 | 0.9511 | 0.0000 |
| **Embedded (RF)** | `avg_glucose_level` | 6 | 1 | 0.9511 | 0.0000 |
| **Wrapper (RFE)** | `heart_disease` | 1 | 1 | 0.9511 | 0.0000 |
| **Combined Top** | `age`, `avg_glucose_level`, `heart_disease` | 12 | 3 | 0.9511 | 0.0000 |

### 3. Dimensionality Reduction

- **Principal Component Analysis (PCA)**:
  - Analyzing cumulative explained variance revealed that **8 components** capture **90.77%** of the variance, while **9 components** capture **97.08%**.
  - Run with 8 components: Best accuracy of **95.11%** at $k=16$.
  - Run with 9 components: Best accuracy of **95.11%** at $k=16$.
  - *Decision*: We select **8 principal components** as it achieves identical accuracy with lower dimensionality.

- **Linear Discriminant Analysis (LDA)**:
  - Since the target variable is binary, LDA yields a maximum of $C - 1 = 1$ dimension.
  - The model projected the 10-dimensional space onto a single discriminant coordinate.
  - Best accuracy: **95.11%** at $k=12$.

---

## Final Performance Comparison

The table below summarizes the performance of the $k\text{NN}$ classifier using the optimal settings from each dimensionality management method:

| Method | Number of Dimensions | Optimal $k$ | Average Accuracy | Standard Deviation |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (All Features)** | 10 | 8 | **0.9512** | 0.00026 |
| **Filter Method (ANOVA)** | 1 | 1 | **0.9511** | 0.00000 |
| **PCA** | 8 | 16 | **0.9511** | 0.00000 |
| **LDA** | 1 | 12 | **0.9511** | 0.00000 |

### Key Conclusions
- **Performance Parity**: All feature selection and dimensionality reduction methods converged on an average accuracy of **95.11%**, which is statistically equivalent to the baseline model's performance (95.12%).
- **Efficiency Champion**: The **Filter Method** using only a single feature (`age`) with $k=1$ is the most optimal setup. It requires no complex linear projections (unlike PCA/LDA) and operates with the absolute minimum computational complexity ($k=1$ lookup on 1 feature).
- **Interpretability**: The Filter and Wrapper methods highlight that `age`, `heart_disease`, and `avg_glucose_level` are key clinical indicators of stroke risk.

---

## Directory Structure

```
stroke-prediction-analysis/
├── README.md                  # Detailed analytical report
├── requirements.txt           # Python dependencies
├── .gitignore                 # Excluded temporary/system files
├── data/                      # Dataset folder
│   ├── healthcare-dataset-stroke-data.csv
│   ├── categorical_features.csv
│   └── numeric_features.csv
├── src/                       # Refactored source scripts
│   ├── knn.py                 # Baseline kNN experiment
│   ├── FilterMethod.py        # ANOVA-based feature selection
│   ├── WrapperMethod.py       # RFE-based feature selection
│   ├── EmbeddedMethod.py      # Random Forest feature selection
│   ├── Feature_Selection.py   # Selected features evaluation
│   ├── PCA.py                 # Principal Component Analysis
│   └── LDA.py                 # Linear Discriminant Analysis

```

---

## Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed. Install the required libraries using:
```bash
pip install -r requirements.txt
```

### Running the Experiments
Navigate to the `src/` directory and execute any of the analysis scripts:

```bash
# Run baseline KNN experiment
python src/knn.py

# Run PCA dimensionality reduction
python src/PCA.py

# Run LDA dimensionality reduction
python src/LDA.py

# Run Filter feature selection
python src/FilterMethod.py
```
Each script will output the accuracy table for different values of $k$ and generate a plot showing accuracy vs. $k$ (with standard deviation error bars).
