# DS-Project

**Comparative Analysis of Machine Learning Algorithms on the UCI Parkinson’s Voice Dataset**  
*Data Science course project*

---

## Table of Contents

1. [Project Title](#project-title)  
2. [Description](#description)  
3. [Dataset](#dataset)  
4. [Model Training & Results](#model-training--results)  
5. [Repository Structure](#repository-structure)  

---

## Project Title

**DS-Parkinsons-ML-Project**

---

## Description

This Data Science course project benchmarks five different machine learning algorithms on the UCI Parkinson’s Voice dataset to determine which approach best distinguishes Parkinson’s‐positive patients from healthy controls. We perform the following steps:

1. **Data Loading & Preprocessing**  
   - Drop identifiers, separate features and labels, and perform a stratified train/test split.

2. **Feature Scaling**  
   - Apply MinMax scaling (range = [−1, 1]) fit on training data only, then transform both train and test sets.

3. **Class‐Imbalance Handling**  
   - Use `class_weight='balanced'` or SMOTE oversampling (for algorithms that don’t accept class weights) to address the ~75% Parkinson’s / ~25% healthy split.

4. **Feature Selection (Optional)**  
   - Filter out features with high pairwise correlation (|ρ| > 0.90) and/or select top 10 features based on tree‐based importances or univariate ANOVA F‐tests.

5. **Model Training & Hyperparameter Tuning**  
   - Train and tune the following classifiers via 5‐fold cross‐validated GridSearchCV (scoring = ROC‐AUC):
     - Decision Tree (DT)  
     - Random Forest (RF)  
     - Logistic Regression (LR)  
     - Support Vector Machine (SVM)  
     - XGBoost (XGB)  

6. **Evaluation**  
   - Evaluate on held‐out test set using accuracy, precision, recall, F₁‐score, and R²‐score for each model.

7. **Comparison & Interpretation**  
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

---
