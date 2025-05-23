# DS-Project

**Comparative Analysis of Machine Learning Algorithms on the UCI Parkinson’s Voice Dataset**  
*Data Science course project*

---


## Description

This Data Science course project benchmarks five different machine learning algorithms on the UCI Parkinson’s Voice dataset to determine which approach best distinguishes Parkinson’s‐positive patients from healthy controls. We perform the following steps:

1. **Data Loading & Preprocessing**  

2. **Feature Scaling**  

3. **Class‐Imbalance Handling**  

4. **Model Training & Hyperparameter Tuning**  
     - Decision Tree (DT)  
     - Random Forest (RF)  
     - Logistic Regression (LR)  
     - Support Vector Machine (SVM)  
     - XGBoost (XGB)  

5. **Comparison & Interpretation**  
   - Compare performance and highlight the top-performing model in terms of accuracy, F₁‐score, recall, precision, and R²‐score.

---

## Dataset

- **Source:** UCI Machine Learning Repository – Parkinson’s Disease Classification  
- **Filename:** `parkinsons_dataset.csv`  
- **Samples:** 195 voice recordings  
  - 147 Parkinson’s (status = 1, ~75%)  
  - 48 Healthy (status = 0, ~25%)  
- **Columns (24 total):**  
  1. `name` – sample identifier (dropped during processing)  
  2. 22 acoustic/motor features (all numeric), for example:  
     - MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)  
     - MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ  
     - Jitter:DDP, MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA  
     - NHR, HNR, DFA, spread1, spread2, D2, PPE  
  3. `status` – target label (1 = Parkinson’s, 0 = Healthy)

---

## Model Training & Results

Each algorithm was tuned via `GridSearchCV` (5‐fold, scoring = ROC‐AUC) on the training set. Below are the final metrics on the held-out test set.

| Metric     | DT      | RF      | LR      | SVM     | XGB     |
|:-----------|:-------:|:-------:|:-------:|:-------:|:-------:|
| Accuracy   | 0.932203 | 0.966102 | 0.830508 | 0.966102 | 0.932203 |
| F1-Score   | 0.920000 | 0.961538 | 0.782609 | 0.960000 | 0.925926 |
| Recall     | 0.884615 | 0.961538 | 0.692308 | 0.923077 | 0.961538 |
| Precision  | 0.958333 | 0.961538 | 0.900000 | 1.000000 | 0.892857 |
| R2-Score   | 0.724942 | 0.862471 | 0.312354 | 0.862471 | 0.724942 |

![Models Comparison](comparison.png)
---
