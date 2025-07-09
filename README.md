# Welcome to the SuperDataScience Community Project!
Welcome to the **SpineScope: Modeling Spinal Health with Biomechanical Data** repository! 🎉

This project is a collaborative initiative brought to you by SuperDataScience, a thriving community dedicated to advancing the fields of data science, machine learning, and AI. We are excited to have you join us in this journey of learning, experimentation, and growth.

To contribute to this project, please follow the guidelines avilable in our [CONTRIBUTING.md](CONTRIBUTING.md) file.

# Project Scope of Works:

## Project Overview
**SpineScope** is a supervised learning project based on biomechanical measurements of orthopedic patients. Participants will build models that predict spinal conditions using features like pelvic tilt, sacral slope, and lumbar angles.

The project is structured in two learning tracks:
- 🟢 **Beginner Track** – Traditional ML workflow with Streamlit deployment
- 🔴 **Advanced Track** – Deep learning for tabular regression/classification, with explainability tools

Link to dataset: https://www.kaggle.com/datasets/uciml/biomechanical-features-of-orthopedic-patients?select=column_2C_weka.csv

## 🟢 Beginner Track: ML-Based Spinal Condition Predictor

### Objectives
#### Exploratory Data Analysis
- Analyze distribution and correlation among biomechanical attributes
- Detect outliers and apply feature scaling or normalization
- Visualize class distributions (if labels are included in dataset)

**Key Questions to Answer**:
- Which features are most strongly correlated with spinal abnormalities?
- Are any features linearly dependent on others (e.g., sacral slope and pelvic incidence)?
- Do biomechanical measurements cluster differently for normal vs. abnormal cases?
- Are there multicollinearity issues that impact modeling?

#### Model Development
- Build and compare classifiers or regressors (depending on target variable): Logistic Regression, Random Forest, Gradient Boosting
- Use **MLflow** to track model metrics and experiment details
- Evaluate performance using accuracy, precision, recall, F1-score (for classification) or RMSE/MAE (for regression)

#### Model Deployment
- Build a **Streamlit app** that accepts six biomechanical inputs and returns a prediction
- Deploy the app to Streamlit Community Cloud

### Technical Requirements
- **Data Handling & Visualization**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `xgboost`, `mlflow`
- **Deployment**: `streamlit`


## 🔴 Advanced Track: Deep Learning for Biomechanical Diagnosis
### Objectives
#### Exploratory Data Analysis
- Perform dimensionality checks and multicollinearity analysis
- Assess distribution skewness and apply transformations if needed
- Explore inter-feature interactions and feature redundancy

**Key Questions to Answer**:
- Can a neural network learn feature interactions better than tree-based models?
- Are non-linear relationships dominant across features like lumbar angle and pelvic incidence?
- Which features contribute most to misclassifications or prediction errors?
- How do feature importances change when learned via embeddings vs. raw input?

#### Model Development
- Build a **Feedforward Neural Network (FFNN)** for classification or regression using **PyTorch** or **TensorFlow**
- Incorporate:
    - Dense layers with dropout and batch normalization
    - ReLU activations and early stopping

- Use **MLflow** for experiment tracking (architecture, learning rate, batch size, etc.)
- Evaluate using standard metrics and residual analysis
- Optional: compare with tree-based methods like LightGBM or CatBoost

#### Explainability
- Use **SHAP values** or **Integrated Gradients** to interpret predictions
- Visualize contribution of each biomechanical attribute per prediction

#### Model Deployment
- Deploy trained DL model in a **Streamlit app**
- Accept user inputs, return predictions, and show SHAP explanation plots
- Host on Streamlit Community Cloud or Hugging Face Spaces

### Technical Requirements
- **Data Handling & Visualization**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Deep Learning**: `tensorflow` or `pytorch`, `mlflow`
- **Explainability**: `shap`, `scikit-learn`, `captum` (for PyTorch)
- **Deployment**: `streamlit`


## Workflow & Timeline (Both Tracks)

| Phase                     | Core Tasks                                                               | Duration      |
| ------------------------- | ------------------------------------------------------------------------ | ------------- |
| **1 · Setup + EDA**       | Setup project repo, clean data, visualize features, answer key questions | **Week 1**    |
| **2 · Model Development** | Train, tune, and evaluate ML/DL models; track with MLflow                | **Weeks 2–4** |
| **3 · Deployment**        | Build and deploy Streamlit app                                           | **Week 5**    |

