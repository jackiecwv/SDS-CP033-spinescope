# 📄 SpineScope – Project Report - 🟢 **Beginner Track**

Welcome to your personal project report!  
Use this file to answer the key reflection questions for each phase of the project. This report is designed to help you think like a data scientist, guide AI tools more effectively, and prepare for real-world job interviews.

---

## ✅ Phase 1: Setup & Exploratory Data Analysis (EDA)

    > Answer the EDA questions provided in the project materials here. Focus on data quality, trends, anomalies, and relationships.

    ### 🔑 Question 1: Which features are most strongly correlated with spinal abnormalities?

    ### 🔑 Question 2: Are any features linearly dependent on others (e.g., sacral slope and pelvic incidence)?

    ### 🔑 Question 3: Do biomechanical measurements cluster differently for normal vs. abnormal cases?

    ### 🔑 Question 4: Are there multicollinearity issues that impact modeling?

---

## ✅ Phase 2: Model Development

> This phase spans 3 weeks. Answer each set of questions weekly as you build, train, evaluate, and improve your models.

---

### 🔍 Week 2: Feature Engineering & Data Preprocessing

#### 🔑 Question 1:
**Which biomechanical features show the strongest relationship with the target (spinal condition), and how did you determine this?**

✏️ **Answer:**

To identify the biomechanical features most strongly associated with the spinal condition, I combined exploratory analysis, correlation metrics, and statistical testing:

'dataset.groupby('class').mean()' allowed me to compare the average values of each biomechanical feature across the spinal condition classes (normal vs. abnormal). Features with large differences between groups hinted at a strong relationship with the condition.

| class | pelvic_incidence | pelvic_tilt | lumbar_lordosis_angle  | sacral_slope | pelvic_radius | degree_spondylolisthesis  |
|-------|------------------|-------------|------------------------|--------------|---------------|---------------------------|
| 0     | 51.685244        | 12.821414   | 43.542605              | 38.86383     | 123.890834    | 2.186572                  |
| 1     | 64.692562        | 19.791111   | 55.925370              | 44.90145     | 115.077713    | 37.777705                 |

**degree_spondylolisthesis** (difference ≈ 35.6) shows the largest difference between normal and abnormal classes, indicating a strong association with the spinal condition.

Other notable features:
**pelvic_incidence** (difference ≈ 13.0)
**lumbar_lordosis_angle** (difference ≈ 12.4)

'dataset.corr()['class'].sort_values(ascending=False)' calculated the correlation between each numerical feature and the binary-encoded target (e.g., 0 = normal, 1 = abnormal).

| Feature                   | Correlation with Class |
|---------------------------|------------------------|
| class                     | 1.000000               |
| degree_spondylolisthesis  | 0.443687               |
| pelvic_incidence          | 0.353336               |
| pelvic_tilt               | 0.326063               |
| lumbar_lordosis_angle     | 0.312484               |
| sacral_slope              | 0.210602               |
| pelvic_radius             | -0.309857              |

Features like **degree_spondylolisthesis** and **pelvic_incidence** had the strongest correlation coefficients, indicating a significant linear relationship with spinal condition.

`boxplot` and `violinplot` helped me visualize how each feature varies across target classes.

![alt text](images/model-dev/boxplot.png)

![alt text](images/model-dev/violinplot.png)

Again, features like degree_spondylolisthesis and pelvic_incidence had the strongest difference.

Statistical Testing (t-tests) helped me to confirm that the observed differences were statistically significant. These t-tests tell you which features are statistically different between classes, making them good candidates for modeling or further analysis

| Feature                       | T-Statistic | P-Value     | Interpretation                  |
| ----------------------------- | ----------- | ----------- | ------------------------------- |
| **degree\_spondylolisthesis** | **-8.69**   | **2.2e-16** | 🔥 **Very strongly associated** |
| **pelvic\_radius**            | **+5.72**   | **2.5e-08** | ✅ Strongly associated          |
| **sacral\_slope**             | **-3.78**   | **0.00019** | ⚠️ Moderately associated        |

✅ Interpretation by Feature:
🔥 1. degree_spondylolisthesis
T-statistic = -8.69: Large difference in means between class 0 and 1.

P-value = 2.2e-16: Extremely small — this is not due to chance.

✅ Most strongly associated with spinal condition.

✅ 2. pelvic_radius
T-statistic = +5.72: Clear difference between class groups.

P-value = 2.5e-08: Very small — statistically significant.

✅ Strong association with spinal condition.

⚠️ 3. sacral_slope
T-statistic = -3.78: Moderate difference between groups.

P-value = 0.00019: Still significant (p < 0.05), but weaker than the others.

⚠️ Moderate association with spinal condition.

---

#### 🔑 Question 2:
Before building any model, what patterns or class imbalances did you observe in the target variable? Will this affect your modeling choices?

✏️ **Answer:**

| class | count |
|-------|-------|
| 1     | 210   |
| 0     | 100   |

Total samples: 310

So, the target variable is imbalanced, with roughly:
- 32% Normal (Class 0)
- 68% Abnormal (Class 1)

An imbalanced dataset like this can lead to a biased models, especially when using models that try to maximize overall accuracy. If we do not choose our model wisely, we will have a dumb model that always predicts class 1 (Abnormal):

Recommendation:

1. **Evaluation Metrics Beyond Accuracy**
Use metrics like:
- Precision, Recall, F1-score,
- Confusion matrix,
- ROC-AUC

2. **Stratified Splits**
Use StratifiedKFold or train_test_split(..., stratify=y) to maintain class proportions during training/testing.

3. **Resampling Techniques**
- Oversample minority class (e.g. SMOTE)
- Undersample majority class
- Or use a balanced-class model (some classifiers like RandomForestClassifier and XGBoost can handle this with a class_weight='balanced' or scale_pos_weight parameter).

4. **Algorithm Choice**
- Tree-based models (Random Forest, XGBoost) are more robust to imbalance.
- Logistic regression or SVM may need class weights.

---

#### 🔑 Question 3:
**Which features appear skewed or contain outliers, and what transformations (if any) did you apply to address them?**  

✏️ **Answer:**

Several biomechanical features in the dataset appear skewed and/or contain outliers, as observed using `.hist()`, `.skew()`, and boxplots:
![alt text](images/model-dev/histogram_skewness.png)
![alt text](images/model-dev/skewness_degree.png)

- **degree_spondylolisthesis** is highly right-skewed, with some extreme outlier values.
- **pelvic_tilt** and **lumbar_lordosis_angle** also show moderate skewness and potential outliers.
- **pelvic_incidence**, **sacral_slope**, and **pelvic_radius** are closer to normal but still show some skew and outliers.

To address these:
- I used boxplots and histograms to visually confirm skewness and outliers.
- For highly skewed features (especially **`degree_spondylolisthesis`**), I applied a log-transform to reduce skewness and the impact of extreme values.
These step helped improve model robustness and performance.

---

#### 🔑 Question 4:
**What scaling method did you apply to your numerical features, and why was it necessary (or not) for the algorithms you plan to use?**  

✏️ **answer:**

For scaling, I applied the **StandardScaler** to all numerical features. This method standardizes features by removing the mean and scaling to unit variance (z-score normalization).

**Reasoning:**
- **Logistic regression** and **distance-based models** (like k-NN, SVM) are sensitive to feature scales, so standardization is necessary to ensure all features contribute equally to the model.
- **Tree-based models** (Random Forest, Gradient Boosting) do not require scaling, but for consistency and to enable fair comparison across different algorithms, I standardized the features for all models.
- Standardization also helps with convergence speed and stability for models that use gradient descent.



---

#### 🔑 Question 5:
    **Did you create any new features that might help distinguish between different spinal conditions? If yes, what are they and what was the reasoning behind them?**  

    💡 **Hint:**  
    Try feature ratios or differences, like `pelvic_incidence - pelvic_tilt`.  
    Use domain insight or trial-and-error to create potentially useful features.

    ✏️ *Your answer here...*

---

### 📆 Week 3: Model Development & Experimentation

#### 🔑 Question 1:
**Which models did you train for predicting spinal conditions, and what were your reasons for choosing them?**  
🎯 *Purpose: Tests model selection reasoning and algorithm familiarity.*

I used Logistic Regression, Random Forest, and XGBoost. But I got more accurary from XGBoost with an CV accuracy score of 0.8806451612903226. I chose these models becasue we have a classification issue and need a classification model.

##### Create a pipeline: scaling + model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBClassifier())
    # ('model', LogisticRegression())
    # ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

##### Use cross-validation (StratifiedKFold for classification)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

##### Evaluate using cross_val_score
scores = cross_val_score(pipeline, X_sds, y, cv=cv, scoring='accuracy')

print("Cross-validation accuracy scores:", scores)
print("Average CV accuracy:", scores.mean())

---

#### 🔑 Question 2:
**What evaluation metrics did you use to assess your models, and how did each model perform?**  
🎯 *Purpose: Tests metric literacy and performance analysis.*
 
I used accuracy, precision, recall, F1-score, confusion matrix.   

LogisticRegression
![alt text](image-4.png)

XGBClassifier
![alt text](image-5.png)

RandomForestClassifier
![alt text](image-3.png)

XGBoost has the highest CV accuracy score
---

#### 🔑 Question 3:
**Do any of your models show signs of overfitting or underfitting? How did you identify this, and what might be the cause?**  
🎯 *Purpose: Tests generalization understanding and diagnostic skills.*

💡 **Hint:**  
Compare training vs. test scores.
Visualize learning curves or residual plots.
Overfitting: great performance on training, poor on test.
Underfitting: poor performance on both.
Suggest potential remedies.

✏️ *Your answer here...*

---

#### 🔑 Question 4:
**Which features contributed the most to your model's predictions, and do the results align with your domain expectations?**  
🎯 *Purpose: Tests model explainability and domain connection.*

💡 **Hint:**  
Use `.coef_` for Logistic Regression or `.feature_importances_` for tree models.
Plot the top features.
Do the top features (e.g., pelvic tilt, lumbar lordosis angle) make clinical sense?

✏️ *Your answer here...*

---

#### 🔑 Question 5:
**How did you use MLflow to track your model experiments, and what comparisons did it help you make more easily?**  
🎯 *Purpose: Tests reproducibility and experiment tracking best practices.*

💡 **Hint:**  
Log model name, hyperparameters, and evaluation metrics.
Include screenshots or links to MLflow runs.
Explain how MLflow helped you select your best model or debug issues.

✏️ *Your answer here...*

---

### 📆 Week 4: Model Tuning

#### 🔑 Question 1:

#### 🔑 Question 2:

#### 🔑 Question 3:

#### 🔑 Question 4:

#### 🔑 Question 5:

---

## ✅ Phase 3: Model Deployment

    > Document your approach to building and deploying the Streamlit app, including design decisions, deployment steps, and challenges.

    ### 🔑 Question 1:

    ### 🔑 Question 2:

    ### 🔑 Question 3:

    ### 🔑 Question 4:

    ### 🔑 Question 5:

---

## ✨ Final Reflections

    > What did you learn from this project? What would you do differently next time? What did AI tools help you with the most?

    ✏️ *Your final thoughts here...*

---
