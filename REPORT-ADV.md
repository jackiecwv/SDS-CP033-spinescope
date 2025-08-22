# 📄 SpineScope – Project Report - 🔴 **Advanced Track**

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

#### 🔑 Question 1:
**Which categorical features are high-cardinality, and how will you encode them for use with embedding layers?**  

💡 **Hint:**  
Use `.nunique()` and `.value_counts()` to inspect cardinality.  
Use `LabelEncoder` or map categories to integer IDs.  
Think about issues like rare categories, overfitting, and embedding size selection.

✏️ *Your answer here...*

---

#### 🔑 Question 1:
**Which biomechanical features are likely to have the most predictive power for classifying spinal conditions, and how did you determine that?**

💡 **Hint:**  
Use `.groupby(target).mean()`, boxplots, or violin plots to see how each feature separates across classes.  
Use correlation matrices or feature importance from an early tree-based model as a sanity check.  
Look for features that consistently differ between classes (e.g., high pelvic tilt in abnormal cases).

✏️ *Your answer here...*

---

#### 🔑 Question 2:
**What numerical features in the dataset needed to be scaled before being input into a neural network, and what scaling method did you choose?**

💡 **Hint:**  
Use `df.describe()` and histograms to evaluate spread and skew.  
Neural networks are sensitive to feature scale.  
Choose between `StandardScaler`, `MinMaxScaler`, or log-transform based on distribution and range.

✏️ *Your answer here...*

---

#### 🔑 Question 3:
**Did you create any new features based on domain knowledge or feature interactions? If yes, what are they and why might they help the model better predict spinal conditions?**

💡 **Hint:**  
Try combining related anatomical angles or calculating ratios/differences (e.g., `pelvic_incidence - sacral_slope`).  
Think: which combinations might reflect spinal misalignment patterns?

✏️ *Your answer here...*

---

#### 🔑 Question 4:
**Which features, if any, did you choose to drop or simplify before modeling, and what was your justification?**

💡 **Hint:**  
Check for highly correlated or constant features using `.corr()` and `.nunique()`.  
Avoid overfitting by removing redundant signals.  
Be cautious about leaking target-related info if any engineered features are overly specific.

✏️ *Your answer here...*

---

#### 🔑 Question 5:
**After preprocessing, what does your final input schema look like (i.e., how many numerical and categorical features)? Are there class imbalance or sparsity issues to be aware of?**

💡 **Hint:**  
Use `.shape`, `.dtypes`, and `.value_counts()` on the target.  
Check if certain features have many near-zero or rare values.  
Look for imbalanced class distributions and think about resampling, class weights, or focal loss.

✏️ *Your answer here...*


---

### ✅ Week 3: Model Development & Experimentation

### 🔑 Question 1:
**What neural network architecture did you implement (input shape, number of hidden layers, activation functions, etc.), and what guided your design choices?**  
🎯 *Purpose: Tests ability to structure an FFNN for classification and explain architectural decisions.*

💡 **Hint:**  
Describe your model layers: e.g., `[Input → Dense(64) → ReLU → Dropout → Dense(32) → ReLU → Output(sigmoid)]`.  
Justify the number of layers/units based on dataset size and complexity.  
Explain why ReLU and sigmoid are appropriate.

✏️ *Your answer here...*

---

### 🔑 Question 2:
**What metrics did you track during training and evaluation (e.g., accuracy, precision, recall, F1-score, AUC), and how did your model perform on the validation/test set?**  
🎯 *Purpose: Tests metric understanding for classification and ability to interpret model quality.*

💡 **Hint:**  
Log and compare metrics across epochs using validation data.  
Plot confusion matrix and/or ROC curve.  
Explain where the model performs well and where it struggles (e.g., false positives/negatives).

✏️ *Your answer here...*

---

### 🔑 Question 3:
**How did the training and validation loss curves evolve during training, and what do they tell you about your model's generalization?**  
🎯 *Purpose: Tests understanding of overfitting/underfitting using learning curves.*

💡 **Hint:**  
Include training plots of loss and accuracy.  
Overfitting → training loss drops, validation loss increases.  
Underfitting → both remain high.  
Mention any regularization techniques used (dropout, early stopping).

✏️ *Your answer here...*

---

### 🔑 Question 4:
**How does your neural network’s performance compare to a traditional baseline (e.g., Logistic Regression or Random Forest), and what insights can you draw from this comparison?**  
🎯 *Purpose: Encourages comparative thinking and understanding model trade-offs.*

💡 **Hint:**  
Train a classical ML model and compare F1, AUC, and confusion matrix.  
Was the neural net better? If so, why (e.g., captured interactions)?  
If not, consider whether your DL model is under-tuned or overfitting.

✏️ *Your answer here...*

---

### 🔑 Question 5:
**What did you log with MLflow (e.g., model configs, metrics, training duration), and how did this help you improve your modeling workflow?**  
🎯 *Purpose: Tests reproducibility and tracking practice in a deep learning workflow.*

💡 **Hint:**  
Log architecture details (e.g., layer sizes, dropout rate, learning rate), metrics per epoch, and confusion matrix screenshots.  
Explain how you used logs to choose the best model or compare runs.

✏️ *Your answer here...*

---

## ✅ Week 4: Model Selection & Hyperparameter Tuning

### 🔑 Question 1:
**What strategies did you use to select your final neural network architecture, and how did you compare different model variants?**  

💡 **Hint:**  
Discuss how you experimented with different numbers of layers, units, activation functions, or regularization.  
Explain what criteria (e.g., validation loss, accuracy, overfitting signs) guided your choices.

✏️ *Your answer here...*

---


### 🔑 Question 2:
**Which hyperparameters did you tune for your neural network, and what search methods did you use to find optimal values?**  

💡 **Hint:**  
List key hyperparameters (e.g., learning rate, batch size, dropout rate, optimizer).  
Describe your tuning approach (manual, grid search, random search, or using tools like Optuna).

✏️ *Your answer here...*

---


### 🔑 Question 3:
**How did hyperparameter tuning affect your model’s performance and generalization?**  

💡 **Hint:**  
Compare metrics before and after tuning.  
Show training/validation curves or tables if possible.  
Discuss any trade-offs (e.g., speed vs. accuracy, overfitting vs. underfitting).

✏️ *Your answer here...*

---


### 🔑 Question 4:
**What validation strategies did you use to ensure robust model selection?**  

💡 **Hint:**  
Describe your use of validation splits, k-fold cross-validation, or stratified sampling.  
Explain why these strategies are important for deep learning models.

✏️ *Your answer here...*

---


### 🔑 Question 5:
**What challenges did you encounter during model selection or hyperparameter tuning, and how did you address them?**  

💡 **Hint:**  
Mention issues like unstable training, long runtimes, or convergence problems.  
Describe any solutions (e.g., early stopping, learning rate schedules, batch normalization).

✏️ *Your answer here...*

---

## ✅ Week 5: Model Deployment

> Document your approach to building and deploying the Streamlit app, including design decisions, deployment steps, and challenges.

### 🔑 Question 1:
**How did you design the user interface of your Streamlit app to support both prediction and model explainability?**  

💡 **Hint:**  
Describe the layout, input fields, and how you present both predictions and SHAP (or other explainability) plots.  
Explain how you made the app accessible and informative for users.

✏️ *Your answer here...*

---


### 🔑 Question 2:
**What steps did you take to integrate your trained deep learning model and explainability tools into the Streamlit app?**  

💡 **Hint:**  
Explain how you loaded the model, handled user input, generated predictions, and computed SHAP or Integrated Gradients explanations.  
Mention any preprocessing applied to user data.

✏️ *Your answer here...*

---


### 🔑 Question 3:
**How did you ensure your app works correctly and robustly, especially with unexpected or invalid user input?**  

💡 **Hint:**  
Discuss input validation, error handling, and any testing strategies you used.

✏️ *Your answer here...*

---


### 🔑 Question 4:
**Describe the process you followed to deploy your Streamlit app online. What platform did you use, and what challenges did you face?**  

💡 **Hint:**  
List deployment steps (e.g., requirements.txt, Streamlit Community Cloud, Hugging Face Spaces).  
Mention any issues and how you resolved them.

✏️ *Your answer here...*

---


### 🔑 Question 5:
**How does your deployed app help users understand the model’s predictions? What explainability features did you include, and why are they important?**  

💡 **Hint:**  
Describe any SHAP plots, feature importance charts, or confidence scores you provided.  
Explain how these features build user trust and support clinical interpretation.---

✏️ *Your answer here...*

---


## ✨ Final Reflections

> What did you learn from this project? What would you do differently next time? What did AI tools help you with the most?

✏️ *Your final thoughts here...*

---
