# 📊 SpineScope Project – Phase 2 Week 4: Additional Model Development

Author: Jackie CW Vescio  
Date: Week of July 29 – August 3, 2025

---

## Objective

This week's focus was to deepen modeling skills by exploring additional classification models beyond Logistic Regression and Random Forest. Emphasis was placed on model training, evaluation, hyperparameter tuning, and visual interpretation of results through performance metrics, confusion matrices, and learning curves.

---

## Models Explored

### 1. **Random Forest (RF)**
- Achieved **87% accuracy**.
- Strong class separation with balanced precision and recall.
- Included:
  - Classification Report
  - Confusion Matrix
  - Feature Importance (table + bar plot)
  - Learning Curve

### 2. **XGBoost (XGB)**
- Achieved **85% accuracy** after tuning.
- Tuned via `GridSearchCV`.
- Logged with MLflow for reproducibility.
- Included:
  - Classification Report
  - Confusion Matrix
  - Model Comparison Bar Plot
  - Learning Curve
  - MLflow Integration

### 3. **Support Vector Classifier (SVC)**
- Initial model reached **85% accuracy**.
- Tuned model also reached **85%**, using a linear kernel.
- Included:
  - Classification Reports (original + tuned)
  - Confusion Matrix
  - Comparison Bar Plot
  - Learning Curve

### 4. **K-Nearest Neighbors (KNN)**
- Achieved **84% accuracy**.
- Required feature scaling due to distance-based logic.
- Included:
  - Classification Report
  - Confusion Matrix
  - Markdown interpretation

---

## Tools and Techniques Used

- `GridSearchCV` for hyperparameter tuning
- `ConfusionMatrixDisplay` and `sns.heatmap` for visualization
- `classification_report` and `accuracy_score` for evaluation
- `learning_curve` for overfitting/underfitting analysis
- `MLflow` for model tracking and parameter logging

---

## File Overview

- `SpineScope_Week4_Phase2_More_Model_Development_JackieCWVescio.ipynb`  
  Contains the full code, evaluations, visualizations, and summaries.

- `images/` folder  
  Stores exported bar plots and visual aids for SVC and XGBoost comparison.

---

## Final Notes

This week’s exploration solidified key model development workflows including:
- Model training and evaluation pipelines
- Comparative analysis using visual metrics
- Tracking experiments with MLflow
- Interpreting performance variations across models
