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

### 🔍 Week 1: Feature Engineering & Data Preprocessing

#### 🔑 Question 1:
Which biomechanical features show the strongest relationship with the target (spinal condition), and how did you determine this?

✏️ Answer:

To identify the biomechanical features most strongly associated with the spinal condition, I combined exploratory analysis, correlation metrics, and statistical testing:

'dataset.groupby('class').mean()' allowed me to compare the average values of each biomechanical feature across the spinal condition classes (normal vs. abnormal). Features with large differences between groups hinted at a strong relationship with the condition.

| class | pelvic_incidence | pelvic_tilt | lumbar_lordosis_angle  | sacral_slope | pelvic_radius | degree_spondylolisthesis  |
|-------|------------------|-------------|------------------------|--------------|---------------|---------------------------|
| 0     | 51.685244        | 12.821414   | 43.542605              | 38.86383     | 123.890834    | 2.186572                  |
| 1     | 64.692562        | 19.791111   | 55.925370              | 44.90145     | 115.077713    | 37.777705                 |

'dataset.corr()['class'].sort_values(ascending=False)' calculated the correlation between each numerical feature and the binary-encoded target (e.g., 0 = normal, 1 = abnormal).

`boxplot` or `violinplot` helped me visualize how each feature varies across target classes.

Statistical Testing (t-tests) helped me to confirm that the observed differences were statistically significant.



pelvic tilt and lumbar lordosis angle showed noticeable differences in their means between normal and abnormal groups.
---

#### 🔑 Question 2:
    Before building any model, what patterns or class imbalances did you observe in the target variable? Will this affect your modeling choices?

    ✏️ Answer:

    class 0 count is 100
    class 1 count is 210

    The class is imbalanced with the count favoring the abnormal class. 

---

#### 🔑 Question 3:
    **Which features appear skewed or contain outliers, and what transformations (if any) did you apply to address them?**  

    💡 **Hint:**  
    Use `.hist()`, `df.skew()`, or boxplots.  
    Try log-transform or standardize features if skewed.  
    Consider z-score or IQR for outlier detection.

    ✏️ *Your answer here...*

---

#### 🔑 Question 4:
    **What scaling method did you apply to your numerical features, and why was it necessary (or not) for the algorithms you plan to use?**  

    💡 **Hint:**  
    Logistic regression and distance-based models require scaling.  
    Tree-based models do not.  
    Use `StandardScaler` or `MinMaxScaler` as needed. Justify your choice.

    ✏️ *Your answer here...*

---

#### 🔑 Question 5:
    **Did you create any new features that might help distinguish between different spinal conditions? If yes, what are they and what was the reasoning behind them?**  

    💡 **Hint:**  
    Try feature ratios or differences, like `pelvic_incidence - pelvic_tilt`.  
    Use domain insight or trial-and-error to create potentially useful features.

    ✏️ *Your answer here...*

---

### 📆 Week 2: Model Development & Experimentation

#### 🔑 Question 1:

#### 🔑 Question 2:

#### 🔑 Question 3:

#### 🔑 Question 4:

#### 🔑 Question 5:

---

### 📆 Week 3: Model Tuning

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
