# Regression Problems: Diabetes Progression Prediction

Code repository and experimental analysis for Assignment 3 of the **Machine Learning** course (7th Semester, Academic Year 2025–2026), Department of Applied Informatics, University of Macedonia.

---

## Problem Description

The objective of this project is to predict disease progression in diabetic patients one year after baseline, based on 10 demographic and clinical/biochemical features[cite: 1].

The analysis uses the built-in dataset from Scikit-learn via `load_diabetes(as_frame=True, scaled=False)`[cite: 1]. The entire experimental pipeline strictly avoids data leakage, applies model explainability through the SHAP library, and follows a rigorous nested cross-validation protocol[cite: 1, 2, 6].

---

## ⚙️ Methodology & Experimental Design

### 1. Validation & Preprocessing Pipeline
* **6-Fold Cross-Validation (`KFold`, `shuffle=True`, fixed `random_state`):** Prevents data leakage and ensures full reproducibility[cite: 2].
* **Min-Max Scaling:** Separate scalers for input features ($X$) and the target variable ($y$), fitted **strictly on the outer training set** of each fold and applied (`transform`) to the test set[cite: 2].
* **Denormalization:** Reversing target predictions back to the original clinical scale using the fold-specific fitted scaler (`inverse_transform`)[cite: 4].
* **Integrity Checks:** Programmatic validation confirming zero sample overlap between train and test splits, and ensuring that each sample serves as a test instance exactly once across the 6 folds[cite: 2].

### 2. Regression Models
Four regression techniques are evaluated[cite: 1]:
1. **Random Forest Regressor** (Mandatory)[cite: 1]
2. **Gaussian Process Regressor** (Mandatory)[cite: 1]
3. **Support Vector Regressor (SVR)** *(or Ridge/Lasso/ElasticNet/XGBoost)*[cite: 1]
4. **Gradient Boosting Regressor** *(or alternative selected model)*[cite: 1]

### 3. Hyperparameter Optimization
* Optimization is performed via **`RandomizedSearchCV`** using inner cross-validation exclusively on the training split of each fold[cite: 3].
* The outer test set remains strictly unseen during hyperparameter tuning, model selection, and early stopping[cite: 3].

### 4. Evaluation Metrics
Models are evaluated on both training and test sets, across both normalized and denormalized scales[cite: 4, 5]:
* **RMSE** (Root Mean Squared Error) – Primary ranking metric[cite: 1, 7]
* **MAE** (Mean Absolute Error)[cite: 1]
* **Max Error**[cite: 1]
* **MAPE** (Mean Absolute Percentage Error) or **$R^2$ Score**[cite: 1, 5]

### 5. Model Explainability (SHAP)
* **Global Explainability:** SHAP summary plots illustrating feature importance and attribution across models and folds[cite: 6].
* **Local Explainability:** SHAP waterfall plots analyzing individual patient predictions for the selected best-performing model[cite: 6].

---

## 📂 Repository Structure

```text
regression-problems-diabetes/
│
├── data/
│   └── results.csv              # Experimental metrics (48 rows: 4 models × 6 folds × 2 sets)
│
├── plots/
│   ├── actual_vs_predicted/     # Actual vs Predicted plots per model and fold
│   └── shap_plots/              # SHAP summary & waterfall plots
│
├── notebook.ipynb               # Google Colab / Jupyter notebook implementation
├── main.py                      # Standalone Python script
├── report.pdf                   # Final written analytical report
├── requirements.txt             # Required Python dependencies
└── README.md                    # Project documentation
