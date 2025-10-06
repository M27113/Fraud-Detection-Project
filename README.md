# Fraud Detection Project

## Overview

This project aims to develop a machine learning model to detect fraudulent transactions in a financial dataset. The goal is to identify suspicious activities and provide actionable insights for prevention strategies.

## Dataset

The dataset used in this project contains transaction records with features such as transaction amount, account balances, and transaction type. It is sourced from [Google Drive](https://drive.google.com/uc?export=download&id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV).

## Objectives

- Perform data cleaning and preprocessing.
- Conduct exploratory data analysis (EDA).
- Engineer relevant features.
- Develop and evaluate a machine learning model.
- Provide insights and recommendations.

---

## Table of Contents
1. [Dataset Description](#dataset-description)  
2. [Objectives](#objectives)  
3. [Exploratory Data Analysis](#exploratory-data-analysis)  
4. [Data Preprocessing](#data-preprocessing)  
5. [Feature Engineering](#feature-engineering)  
6. [Model Development](#model-development)  
7. [Model Evaluation](#model-evaluation)  
8. [Key Findings](#key-findings)  
9. [Prevention Measures & Monitoring](#prevention-measures--monitoring)  
10. [Outputs & Results](#outputs--results)  
11. [How to Run](#how-to-run)  
12. [Acknowledgements](#acknowledgements)

---

## Dataset Description
The dataset contains transaction records for 30 days (744 hourly steps), with the following key columns:  
- Dataset Summary
  - Total Rows: 6,362,620
  
  - Total Columns: 10
- `step`: Time step (1 step = 1 hour)  
- `type`: Transaction type (`CASH-IN`, `CASH-OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`)  
- `amount`: Transaction amount in local currency  
- `nameOrig` / `nameDest`: Sender and recipient accounts  
- `oldbalanceOrg` / `newbalanceOrig`: Sender balances before and after transaction  
- `oldbalanceDest` / `newbalanceDest`: Recipient balances (not available for merchants)  
- `isFraud`: Target variable indicating fraudulent transactions  
- `isFlaggedFraud`: Flags illegal transfers >200,000

**Data Source:** 

You can access the project dataset here: [Dataset](https://drive.google.com/uc?export=download&id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV)


---

## Objectives
- Identify fraudulent transactions in real-time using machine learning.  
- Understand key factors that predict fraud.  
- Provide actionable recommendations for preventing fraud.  

---

## Exploratory Data Analysis
Key visualizations generated during EDA:

1. **Fraud vs Non-Fraud Transactions:**  
   ![Fraud Distribution](outputs/fraud_distribution.png)

2. **Correlation Matrix:**  
   ![Correlation Matrix](outputs/correlation_matrix.png)

---

## Data Preprocessing
- Missing values imputed for numeric (median) and categorical (mode) columns.  
- Outliers removed using IQR method (except for critical columns like `amount` and `isFraud`).  
- Categorical features encoded (one-hot).  
- Numeric features scaled for consistency (optional for tree-based models).  

---

## Feature Engineering
- Non-predictive columns dropped: `nameOrig`, `nameDest`, `isFlaggedFraud`.  
- Key engineered features: `transaction type` encoding, balance difference calculations.  
- Top predictive features selected based on domain knowledge and correlation analysis.  

---

## Model Development
- **Model Used:** XGBoost Classifier (gradient boosting decision trees)  
- **Hyperparameters:**  
  - `n_estimators=100`  
  - `max_depth=5`  
  - `learning_rate=0.1`  
  - `scale_pos_weight` for handling class imbalance  
- **Training Method:** Stratified train-test split (80-20) to maintain fraud ratio.  

---

## Model Evaluation

- **Classification Report:** Precision, Recall, F1-score for fraud detection.  
- **ROC-AUC Score:** Measures the model's ability to distinguish between classes.  
- **Confusion Matrix:**  
  ![Confusion Matrix](outputs/confusion_matrix.png)

- **Top 10 Feature Importances:**  
  ![Feature Importance](outputs/feature_importance.png)
  
-  **Precision-Recall Curve & PR-AUC:**
  
   ![Precision-Recall Curve](outputs/precision_recall.png)

-  **Model Performance**  
     - Classification Report:
    ```
                    precision    recall  f1-score   support
      
                 0     1.0000    0.9913    0.9956   1270881
                 1     0.1291    0.9988    0.2287      1643
       
          accuracy                         0.9913   1272524
         macro avg     0.5646    0.9950    0.6122   1272524
      weighted avg     0.9989    0.9913    0.9946   1272524
   
  ```
      
      - ROC-AUC Score: 0.9997
```
---

## Key Findings

- **Top Predictors of Fraud:**  
  - `amount`, `oldbalanceOrg`, `newbalanceOrig`  
  - `type_TRANSFER`, `type_CASH_OUT`  
- These features align with real-world patterns: large transfers, rapid balance depletion, and specific transaction types indicate higher risk.  

---

## Prevention Measures & Monitoring

- Implement **real-time transaction monitoring** and alerts.  
- Set **threshold-based controls** for large transactions.  
- Retrain models periodically to capture evolving fraud patterns.  
- Track KPIs: fraud detection rate, false positives, precision, recall, and conduct pre-post analysis.  
- Use A/B testing to validate the effectiveness of preventive actions.

---

## Outputs & Results
All outputs are saved in the `outputs/` folder:  
- `fraud_distribution.png` – Fraud vs Non-Fraud counts  
- `correlation_matrix.png` – Feature correlations  
- `confusion_matrix.png` – Model predictions vs actual  
- `feature_importance.png` – Top predictive features  
- `xgb_fraud_model.pkl` – Trained XGBoost model  
-  `precision_recall_curve.png` –  Precision vs Recall
---

## How to Run
1. Clone the repository and install the required dependencies:

   ```bash
   git clone https://github.com/M27113/Fraud-Detection-Project.git
   cd Fraud-Detection-Project
   pip install -r requirements.txt


2. Usage

- Launch the Jupyter Notebook:
     ```bash
     jupyter notebook Fraud_Detection.ipynb



3. Outputs

- Generated outputs are saved in the outputs/ directory, including:

   - Visualizations

   - Trained models

   - Evaluation metrics

## Acknowledgements

- Dataset sourced from [Google Drive](https://drive.google.com/uc?export=download&id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV).

- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

- Inspired by best practices for fraud detection in financial institutions
