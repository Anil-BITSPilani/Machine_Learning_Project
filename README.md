# Machine_Learning_Project
Machine Learning Project for Credit Risk analysis for financial institution
# Bank Marketing Term Deposit Predictor

An end-to-end machine learning project developed to predict whether a retail banking customer will subscribe to a term deposit product following targeted telemarketing campaigns. This project utilizes structural preprocessing pipelines, handles severe class imbalances, and compares **Eager Learning (Decision Trees)** with **Lazy Learning (k-Nearest Neighbors)** approaches.

## 📊 Dataset & Domain Overview
* **Source:** UCI Bank Marketing Dataset (via Kaggle API: `adityamhaske/bank-marketing-dataset`).
* **Domain:** Retail banking, customer acquisition, and campaign analytics.
* **Objective:** Binary classification (`yes` / `no`) predicting prospective client subscription rates.
* **Target Distribution:** 88.5% `no` / 11.5% `yes` (highly skewed minority class).

## 🛠️ Tech Stack & Environment
* **Language:** Python 3.12
* **Platform:** Google Colab Interactive Runtime
* **Core Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib

## ⚙️ Data Engineering & Leakage Protection
1. **Sanity Check:** Verified 0 missing cells and 0 duplicate rows across 4,521 baseline records.
2. **Feature Engineering:**
   * Parsed `education` as a structured ordinal mapping (`unknown` → 0, `primary` → 1, `secondary` → 2, `tertiary` → 3).
   * Encoded `month` string features into cyclical temporal integers (1 to 12).
   * Multi-collinearity management using `pd.get_dummies(drop_first=True)`.
3. **Data Leakage Mitigation:** Dropped the `duration` feature entirely. Call duration is only known after a call finishes, artificially inflating baseline validation performance during evaluation.
4. **Distance Standardization:** Handled scale-sensitive metrics for geometric distance evaluations using `StandardScaler` to prevent unbonded features (like `balance`) from overwhelming the model.

## 📈 Evaluation Matrix Summary

The model splits data into an 80/20 stratified holdout format (3,616 training rows, 905 validation test rows) to strictly preserve target distributions.

| Architecture Evaluation Model | Test Accuracy | Precision (Class 1) | Recall (Class 1) | Key Hyperparameters / Core Adjustments |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline Decision Tree** | 81.66% | 21.30% | 22.00% | `Unpruned` (Overfits noise under class imbalance) |
| **Optimized Pruned Tree** | **89.28%** | *Improved* | *Improved* | `max_depth=8`, `min_samples_split=10`, `min_samples_leaf=19` |
| **k-NN (Overfitted Baseline)** | 81.99% | 20.79% | 20.19% | `k = 1` (High variance, over-indexes on nearest neighbors) |
| **k-NN (Optimal Standardized Range)**| 88.62% | **54.55%** | 5.77% | `k = 7` (Optimal smoothing boundary regularization) |

## 🧠 Model Interpretability

### Extracted Boolean Rule Paths
The tuned Decision Tree was successfully mapped into human-readable IF-THEN paths for business logic integration:
* **Rule Path 1 (Predict 'no'):** IF `poutcome_success` ≤ 0.5000 AND `age` ≤ 1.8279 AND `contact_unknown` ≤ 0.5000 AND `balance` ≤ 0.1668 → **Predict 'no'**
* **Rule Path 2 (Predict 'yes'):** IF `poutcome_success` > 0.5000 AND `day` > 0.0709 → **Predict 'yes'**

## 🚀 How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd bank-marketing-predictor
   ```

2. **Run in Google Colab:**
   Upload the project notebook to Google Colab and ensure your Kaggle credentials are added to the Colab Secrets panel as environment variables to dynamically trigger the automated data ingestion pipeline.
