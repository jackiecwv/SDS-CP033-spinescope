# 📄 SpineScope – Project Report - 🔴 **Advanced Track**

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

### 📆 Week 1: Feature Engineering & Data Preprocessing

#### 🔑 Question 1:
**Which categorical features are high-cardinality, and how will you encode them for use with embedding layers?**  

💡 **Hint:**  
Use `.nunique()` and `.value_counts()` to inspect cardinality.  
Use `LabelEncoder` or map categories to integer IDs.  
Think about issues like rare categories, overfitting, and embedding size selection.

✏️ *Your answer here...*
class: 3 unique values
class
Spondylolisthesis    150
Normal               100
Hernia                60
Name: count, dtype: int64
class mapped to integer IDs for embedding.
Embedding size for 'class': 2

High-cardinality features are those with many unique values (e.g., >10).
- Integer encoding is required for embedding layers.
- Grouping rare categories can help prevent overfitting and reduce embedding size.

**`class: 3 unique values` means the 'class' column has 3 different categories: Spondylolisthesis, Normal, and Hernia.**
**The value counts show how many samples belong to each class.**
**`class mapped to integer IDs for embedding.` means each class label was converted to a unique integer (e.g., Spondylolisthesis=2, Normal=1, Hernia=0).**
**`Embedding size for 'class': 2` means that, following the rule-of-thumb (`min(50, (num_categories + 1) // 2)`), the recommended embedding vector size for this feature is 2.**

- None of the classes in the 'class' column have high cardinality.
- The 'class' feature has only 3 unique values: Spondylolisthesis, Normal, and Hernia.
- High cardinality typically refers to features with many unique categories (e.g., >10), which is not the case here.
---

#### 🔑 Question 1:
**Which biomechanical features are likely to have the most predictive power for classifying spinal conditions, and how did you determine that?**

- Features such as **pelvic_incidence**, **lumbar_lordosis_angle**, and **pelvic_tilt** are likely to have the most predictive power.
- This was determined by:
  - Examining boxplots for each feature by class, which showed clear separation in medians and ranges for these features between normal and abnormal cases.
  - Reviewing the correlation matrix, which indicated strong relationships between these features and the target.
  -  In summary, features that consistently differ between classes and show strong separation in visualizations or model importances are most predictive for spinal condition classification.

---

#### 🔑 Question 2:
**What numerical features in the dataset needed to be scaled before being input into a neural network, and what scaling method did you choose?**

All numerical features are scaled before input into the neural network, as neural networks are sensitive to feature scale.  
- I used `df.describe()` and histograms to check the spread and skew of each feature.
- Based on the distributions, I chose `StandardScaler` (z-score normalization) to standardize the features to zero mean and unit variance.
- This ensures all features contribute equally during training and helps the model converge faster and more reliably.
- The advantage of using `StandardScaler` is that it standardizes features to have zero mean and unit variance.
- This helps neural networks and many machine learning algorithms converge faster and more reliably, prevents features with larger scales from dominating the learning process, and improves numerical stability during optimization.

- `MinMaxScaler` would have scaled each numerical feature to a fixed range, typically [0, 1]. This is useful when features have different ranges and you want to preserve the shape of the original distribution.
- A log-transform would have reduced the skewness of features with long tails or large outliers, making their distributions more symmetric and potentially improving model performance for highly skewed data.

- I found that most numerical features had approximately symmetric or moderately skewed distributions, with values spread around their means and no extreme outliers or long tails.
- Because of this, `StandardScaler` (z-score normalization) was appropriate, as it centers and scales features without distorting their distribution.
- If the features had been highly skewed or had large outliers, a log-transform or `MinMaxScaler` might have been more suitable.

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
I did not drop any features. In deep learning I learnt that neural networks are very sensitive and hence any features could help increase the performance. Perhaps during testing, we can fine tune by testing with some less features
---

#### 🔑 Question 5:
**After preprocessing, what does your final input schema look like (i.e., how many numerical and categorical features)? Are there class imbalance or sparsity issues to be aware of?**

- After preprocessing, the final input schema consists of **6 numerical features** (all biomechanical measurements) and **no high-cardinality categorical features**.
- All features are continuous and have been scaled using `StandardScaler`.
- The target variable (`class`) has 3 classes: Spondylolisthesis, Normal, and Hernia.
- Using `.value_counts()` on the target shows some class imbalance:
  - Spondylolisthesis: 150 samples
  - Normal: 100 samples
  - Hernia: 60 samples
- There are no features with excessive sparsity or many near-zero values.
- **Note:** The class imbalance (especially fewer Hernia cases) may require handling via class weights, resampling, or specialized loss functions during model training.

---

### 📆 Week 2: Model Development & Experimentation

### 🔑 Question 1:
### What neural network architecture did you implement (input shape, number of hidden layers, activation functions, etc.), and what guided your design choices?

[Input → Dense(6) → ReLU → Dropout → Dense(36) (4 Hidden Layers) → ReLU → Output(softmax)]. Use of softmax as we have For multiclass classification (3 classes)
Use loss='sparse_categorical_crossentropy' (if your labels are integers, as with LabelEncoder).
Change your output layer to units=3 and activation='softmax'.

### 🔑 Question 2: What metrics did you track during training and evaluation (e.g., accuracy, precision, recall, F1-score, AUC), and how did your model perform on the validation/test set?
The model’s accuracy and confusion matrix were tracked. For deeper insight, add precision, recall, and F1-score. The confusion matrix shows which classes are most often confused. means:

All 11 Hernia samples were predicted as Spondylolisthesis.
All 20 Normal samples were predicted as Spondylolisthesis.
All 31 Spondylolisthesis samples were correctly predicted.
Accuracy: 0.5 means the model got 50% of the test samples correct (all Spondylolisthesis), but failed to predict Hernia or Normal at all.

Summary:
The model is only predicting the Spondylolisthesis class for every sample, ignoring the other classes. This is a sign of poor model performance and likely class imbalance.

further: ALso tracked precision, recall, and F1-score using classification_report from sklearn.metrics.
These metrics show that the model is only predicting the "Spondylolisthesis" class for all test samples:

Precision, recall, and F1-score for Hernia and Normal are 0.00: The model did not correctly predict any samples for these classes.
Recall for Spondylolisthesis is 1.00: All actual Spondylolisthesis samples were predicted as Spondylolisthesis.
Precision for Spondylolisthesis is 0.50: Only half of the predicted Spondylolisthesis samples were actually correct (the rest should have been Hernia or Normal).
Accuracy is 0.50: The model got 50% of the test samples correct, but only for one class.
Macro and weighted averages are low: This reflects poor performance across all classes except Spondylolisthesis.
Interpretation:
The model is suffering from severe class imbalance or poor generalization. It fails to identify Hernia and Normal cases, predicting only Spondylolisthesis. This means the model is not useful for distinguishing between all classes and needs improvement (e.g., better balancing, tuning, or feature engineering).
### 🔑 Question 3: How did the training and validation loss curves evolve during training, and what do they tell you about your model's generalization? 🎯 Purpose: Tests understanding of overfitting/underfitting using learning curves.
During training, the **training loss** steadily decreased, indicating that the model was learning to fit the training data. However, the **validation loss** remained flat or even increased after a few epochs. This pattern suggests that the model was **overfitting**: it learned the training data well but failed to generalize to unseen data.

- If the validation loss is much higher than the training loss, it means the model is memorizing the training set and not learning patterns that generalize.
- If both losses decrease together, the model is generalizing well.
- In this case, the gap between training and validation loss, combined with poor test metrics (accuracy, precision, recall), confirms that the model does **not generalize well** and struggles to predict minority classes.

**Conclusion:**  
The loss curves indicate overfitting and poor generalization. To improve, consider regularization, dropout, better class balancing (e.g., SMOTE), or tuning the model architecture.

### 🔑 Question 4:

### 🔑 Question 5: What did you log with MLflow (e.g., model configs, metrics, training duration), and how did this help you improve your modeling workflow? 🎯 Purpose: Tests reproducibility and tracking practice in a deep learning workflow.
With MLflow, I logged the following:
After adding MLflow logging to the code, the following were tracked:

- **Model configurations**: architecture details (number of layers, activation functions, optimizer, loss function, batch size, epochs).
- **Metrics**: training and validation accuracy, loss, precision, recall, F1-score (from the last epoch).
- **Training duration**: total time taken for each run (automatically tracked by MLflow).
- **Artifacts**: trained model files, plots of loss/accuracy curves, confusion matrix images, and classification report JSON files.
#### Ml flow plot
When the confusion matrix plot shows most predictions at 1.5 to 2.5 on the x-axis (Predicted), it means that almost all test samples were classified as the class with index 2 (likely "Spondylolisthesis").

Interpretation:

The x-axis represents predicted class labels (e.g., 0 = Hernia, 1 = Normal, 2 = Spondylolisthesis).
If the color intensity is highest between 1.5 and 2.5, it means the model predicted class 2 for nearly every sample.
This confirms the model is not distinguishing between classes and is only predicting one class, matching your earlier metrics and confusion matrix.
Summary:
The plot visually confirms severe class imbalance or poor generalization: the model predicts only "Spondylolisthesis" for all inputs.

**How this helped:**
- Enabled easy comparison of different experiments and hyperparameters.
- Made results reproducible and traceable.
- Helped identify which configurations led to better generalization and performance.
- Facilitated rollback to previous best models and streamlined collaboration.

Overall, MLflow improved experiment tracking, reproducibility, and model selection in the workflow.
---
#### NB: a second test Input 12 units, 2 hidden layers with 64 dense produced worse results
If the confusion matrix plot shows most predictions at 0.5 to 1.5 on the x-axis (Predicted), it means the model is predicting class with index 1 (likely "Normal") for nearly every test sample.

Interpretation:

The x-axis represents predicted class labels (0 = Hernia, 1 = Normal, 2 = Spondylolisthesis).
High color intensity between 0.5 and 1.5 means most predictions are for class 1.
This indicates the model is only predicting "Normal" for all inputs, ignoring the other classes.
Summary:
Just like predicting only class 2, this shows poor generalization and class imbalance, but now the model is biased toward "Normal" instead of "Spondylolisthesis".


### 📆 Week 3: Model Tuning

### 🔑 Question 1:

### 🔑 Question 2:

### 🔑 Question 3:

### 🔑 Question 4:

### 🔑 Question 5:

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
