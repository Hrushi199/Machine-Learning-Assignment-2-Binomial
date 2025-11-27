# Predicting Founder Retention in Startups

**Team:** Unsupervised Learners
* Sawant Hrushikesh Rahul (IMT2023619)
* Akshat Mittal (IMT2023606)

---

## 1. Project Overview

This project focuses on building a supervised machine learning classification model to predict the retention status of startup founders. Founder turnover is a critical signal in the startup ecosystem; the departure of a founder can indicate instability, impact investor confidence, and disrupt operations.

The objective is to analyze demographic, operational, and psychological features to determine whether a founder is likely to have **Left** or **Stayed** with their venture. This predictive capability is vital for Venture Capitalists and HR analytics to identify high-risk ventures early.

## 2. Problem Statement

* **Problem Type:** Supervised Binary Classification
* **Target Variable:** `retention_status` (Classes: `Left`, `Stayed`)
* **Evaluation Metric:** The primary metric for model optimization and selection is the **Macro F1 Score** (to handle potential class imbalances), alongside **Accuracy** and **ROC-AUC**.

## 3. Dataset

The dataset consists of 49 columns capturing diverse attributes of the founders and their startups.
* **`train.csv`**: Contains the training samples with features like `monthly_revenue_generated`, `team_size_category`, `founder_age`, and the target `retention_status`.
* **`test.csv`**: Contains the test samples for which predictions must be generated.

## 4. Methodology

Our approach utilizes a distinct preprocessing pipeline for different model architectures, followed by rigorous hyperparameter tuning using **Optuna**.

### 4.1. Preprocessing & Feature Engineering

1.  **Imputation:**
    * **Tree Models (CatBoost/XGBoost):** Missing categorical values were filled with an explicit `'Unknown'` token to allow the models to learn from "missingness." Numerical values were capped at the 99th percentile or Log-transformed (for XGBoost).
    * **Distance Models (SVM/NN):** Missing values were imputed using the median.

2.  **Encoding Strategy:**
    * **CatBoost:** Utilized native support for categorical features (Ordered Boosting).
    * **XGBoost & SVM:** Applied One-Hot Encoding (`pd.get_dummies`) for nominal features and ordinal mapping for ranked features.
    * **Neural Network:** Used **Entity Embeddings** to learn dense vector representations for high-cardinality categorical features.

3.  **Feature Engineering:**
    * **Ratios:** Created features like `revenue_per_year` and `funding_velocity`.
    * **Interactions:** Combined features to capture complexity, such as `stage_tenure_interaction`.
    * **Binning:** Discretized continuous variables like revenue into bins to handle outliers.

### 4.2. Models and Architectures

We implemented and tuned four distinct architectures:

1.  **Linear SVM (Calibrated):** A `LinearSVC` wrapped in `CalibratedClassifierCV`. It utilizes Platt Scaling to output probabilities, proving that the decision boundary has strong linear components.
2.  **CatBoost Classifier (Winning Model):** Optimized for categorical data using symmetric trees and ordered target statistics. Tuned via Optuna (Depth=4, L2=6).
3.  **XGBoost Classifier:** A gradient-boosted tree model using `log1p` transformed revenue data and heavy L1/L2 regularization.
4.  **Neural Network ("GoatedMLP"):** A custom PyTorch architecture featuring:
    * Entity Embedding layers for categorical inputs.
    * A dense MLP structure (512 $\to$ 128 $\to$ 128).
    * Batch Normalization, SiLU activation, and high Dropout (~0.5).

## 5. Results

The **CatBoost Classifier** achieved the highest performance on the validation set, demonstrating superior handling of categorical features without the sparsity introduced by One-Hot Encoding.

| Model | Macro F1 Score | Key Characteristic |
| :--- | :--- | :--- |
| **CatBoost** | **0.753** | **Ordered Boosting, Native Categorical Support** |
| Linear SVM | 0.749 | Calibrated Probabilities, High-Dim Linearity |
| XGBoost | 0.746 | Log-Transformed Revenue, Gradient Boosting |
| Neural Network | 0.744 | Entity Embeddings, Custom PyTorch Architecture |

## 6. File Descriptions

* **`train.csv` / `test.csv`**: The source datasets.
* **`preprocess.ipynb`**: Contains the EDA (Exploratory Data Analysis) and initial data cleaning experiments.
* **`svm and catboost.ipynb`**: Implementation of the Linear SVM (with calibration) and the CatBoost pipeline. This notebook generates the winning submission.
* **`xgboost.ipynb`**: Implementation of the XGBoost model with Log-transforms and Optuna tuning.
* **`neural_networks.ipynb`**: The PyTorch implementation of the custom "GoatedMLP" model, including the `FounderDataset` class and training loop.
* **`submission_optuna.csv`**: (Generated) The final submission file from the best performing model.

## 7. How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Hrushi199/Machine-Learning-Assignment-2-Binomial.git](https://github.com/Hrushi199/Machine-Learning-Assignment-2-Binomial.git)
    cd Machine-Learning-Assignment-2-Binomial
    ```

2.  **Install dependencies:**
    Ensure you have Python 3.10+ and the following libraries:
    ```bash
    pip install pandas numpy scikit-learn xgboost catboost torch optuna matplotlib
    ```

3.  **Run the Models:**
    * To reproduce the **Winning Model (CatBoost)** and the SVM baseline, run **`svm and catboost.ipynb`**.
    * To train the Deep Learning model, run **`neural_networks.ipynb`**.
    * To run the Gradient Boosting alternative, run **`xgboost.ipynb`**.
