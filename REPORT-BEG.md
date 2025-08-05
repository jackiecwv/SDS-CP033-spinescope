# 📄 SpineScope – Project Report - 🟢 **Beginner Track**

Welcome to your personal project report!  
Use this file to answer the key reflection questions for each phase of the project. This report is designed to help you think like a data scientist, guide AI tools more effectively, and prepare for real-world job interviews.

---

## ✅ Week 1: Setup & Exploratory Data Analysis (EDA)

> Answer the EDA questions provided in the project materials here. Focus on data quality, trends, anomalies, and relationships.

### 🔑 Question 1: Which features are most strongly correlated with spinal abnormalities?

### 🔑 Question 2: Are any features linearly dependent on others (e.g., sacral slope and pelvic incidence)?

### 🔑 Question 3: Do biomechanical measurements cluster differently for normal vs. abnormal cases?

### 🔑 Question 4: Are there multicollinearity issues that impact modeling?

---

## ✅ Week 2: Feature Engineering & Data Preprocessing

### 🔑 Question 1:
**Which biomechanical features show the strongest relationship with the target (spinal condition), and how did you determine this?**

💡 **Hint:**  
Use `.groupby(target).mean()`, `.corr()`, and plots like `boxplot` or `violinplot` to inspect feature separation by class.  
Consider using statistical tests (e.g., ANOVA or t-tests) to validate separation.

✏️ *Your answer here...*

---

### 🔑 Question 2:
**Before building any model, what patterns or class imbalances did you observe in the target variable? Will this affect your modeling choices?**  

💡 **Hint:**  
Use `.value_counts()` or bar plots on the target column.  
If one class dominates, consider techniques like class weights or stratified sampling.

✏️ *Your answer here...*

---

### 🔑 Question 3:
**Which features appear skewed or contain outliers, and what transformations (if any) did you apply to address them?**  

💡 **Hint:**  
Use `.hist()`, `df.skew()`, or boxplots.  
Try log-transform or standardize features if skewed.  
Consider z-score or IQR for outlier detection.

✏️ *Your answer here...*

---

### 🔑 Question 4:
**What scaling method did you apply to your numerical features, and why was it necessary (or not) for the algorithms you plan to use?**  

💡 **Hint:**  
Logistic regression and distance-based models require scaling.  
Tree-based models do not.  
Use `StandardScaler` or `MinMaxScaler` as needed. Justify your choice.

✏️ *Your answer here...*

---

### 🔑 Question 5:
**Did you create any new features that might help distinguish between different spinal conditions? If yes, what are they and what was the reasoning behind them?**  

💡 **Hint:**  
Try feature ratios or differences, like `pelvic_incidence - pelvic_tilt`.  
Use domain insight or trial-and-error to create potentially useful features.

✏️ *Your answer here...*

---

## ✅ Week 3: Model Development & Experimentation

### 🔑 Question 1:
**Which models did you train for predicting spinal conditions, and what were your reasons for choosing them?**  
🎯 *Purpose: Tests model selection reasoning and algorithm familiarity.*

💡 **Hint:**  
Discuss models like Logistic Regression, Random Forest, and XGBoost.  
Explain how each model fits the structure of your data and whether your task is classification (e.g., Normal vs Abnormal) or regression (if predicting continuous values).  
Include code snippets showing model training.

✏️ *Your answer here...*

---

### 🔑 Question 2:
**What evaluation metrics did you use to assess your models, and how did each model perform?**  
🎯 *Purpose: Tests metric literacy and performance analysis.*

💡 **Hint:**  
For classification: accuracy, precision, recall, F1-score, confusion matrix.   
Present results as a table or bar plot.  
Comment on which model had the best generalization on the test set.

✏️ *Your answer here...*

---

### 🔑 Question 3:
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

### 🔑 Question 4:
**Which features contributed the most to your model's predictions, and do the results align with your domain expectations?**  
🎯 *Purpose: Tests model explainability and domain connection.*

💡 **Hint:**  
Use `.coef_` for Logistic Regression or `.feature_importances_` for tree models.
Plot the top features.
Do the top features (e.g., pelvic tilt, lumbar lordosis angle) make clinical sense?

✏️ *Your answer here...*

---

### 🔑 Question 5:
**How did you use MLflow to track your model experiments, and what comparisons did it help you make more easily?**  
🎯 *Purpose: Tests reproducibility and experiment tracking best practices.*

💡 **Hint:**  
Log model name, hyperparameters, and evaluation metrics.
Include screenshots or links to MLflow runs.
Explain how MLflow helped you select your best model or debug issues.

✏️ *Your answer here...*

---

## ✅ Week 4: Model Selection & Hyperparameter Tuning

### 🔑 Question 1:
**How did you select your final model, and what criteria did you use to compare different algorithms?**  

💡 **Hint:**  
Discuss how you compared models (e.g., validation scores, cross-validation, interpretability, speed).  
Explain why you chose your final model for deployment.

✏️ *Your answer here...*

---

### 🔑 Question 2:
**What hyperparameters did you tune for your best-performing model, and how did you search for the optimal values?**  

💡 **Hint:**  
List the key hyperparameters (e.g., number of trees, max depth, learning rate).  
Describe your tuning method (e.g., GridSearchCV, RandomizedSearchCV, manual tuning).

✏️ *Your answer here...*

---

### 🔑 Question 3:
**How did hyperparameter tuning affect your model’s performance?**  

💡 **Hint:**  
Compare metrics before and after tuning.  
Show a table or plot if possible.  
Discuss any trade-offs (e.g., accuracy vs. overfitting).

✏️ *Your answer here...*

---

### 🔑 Question 4:
**Did you use cross-validation or other techniques to ensure your results are robust?**  

💡 **Hint:**  
Explain your validation strategy (e.g., k-fold, stratified split).  
Discuss why this is important for model selection.

✏️ *Your answer here...*

---

### 🔑 Question 5:
**What challenges did you encounter during model selection or tuning, and how did you address them?**  

💡 **Hint:**  
Mention issues like overfitting, long training times, or inconsistent results.  
Describe any solutions or adjustments you made.

✏️ *Your answer here...*

---

## ✅ Week 5: Model Deployment

### 🔑 Question 1:
**How did you design the user interface of your Streamlit app to make it intuitive for users?**  

💡 **Hint:**  
Describe the layout, input fields, and any visualizations you included.  
Explain how you made the app user-friendly.

✏️ *Your answer here...*

---

### 🔑 Question 2:
**What steps did you take to integrate your trained model into the Streamlit app?**  

💡 **Hint:**  
Explain how you loaded the model, handled user input, and generated predictions.  
Mention any preprocessing applied to user data.

✏️ *Your answer here...*

---

### 🔑 Question 3:
**How did you ensure your app works correctly and handles invalid or unexpected user input?**  

💡 **Hint:**  
Discuss input validation, error handling, and testing strategies.

✏️ *Your answer here...*

---

### 🔑 Question 4:
**Describe the process you followed to deploy your Streamlit app online. What platform did you use, and what challenges did you face?**  

💡 **Hint:**  
List deployment steps (e.g., requirements.txt, Streamlit Community Cloud).  
Mention any issues and how you resolved them.

✏️ *Your answer here...*

---

### 🔑 Question 5:
**How does your deployed app help users understand the model’s predictions? Did you include any explanations or visual feedback?**  

💡 **Hint:**  
Describe any feature importance plots, confidence scores, or explanations you provided.  
Explain why this is important for user trust.

✏️ *Your answer here...*

---

## ✨ Final Reflections

> What did you learn from this project? What would you do differently next time? What did AI tools help you with the most?

✏️ *Your final thoughts here...*

---
