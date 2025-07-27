# 📄 SpineScope – Project Report - 🔴 **Advanced Track**

Welcome to your personal project report!  
Use this file to answer the key reflection questions for each phase of the project. This report is designed to help you think like a data scientist, guide AI tools more effectively, and prepare for real-world job interviews.

---

## ✅ Phase 1: Setup & Exploratory Data Analysis (EDA)

> Answer the EDA questions provided in the project materials here. Focus on data quality, trends, anomalies, and relationships.

### 🔑 Question 1: Which features are most strongly correlated with spinal abnormalities?
Feature Interactions Analysis:
-----------------------------------
                    Feature  F_Statistic  P_Value  Effect_Size  Significant
0          pelvic_incidence    98.539709      0.0     0.242984         True
1               pelvic_tilt    21.299194      0.0     0.064877         True
2     lumbar_lordosis_angle   114.982840      0.0     0.272482         True
3              sacral_slope    89.643953      0.0     0.226006         True
4             pelvic_radius    16.866935      0.0     0.052080         True
5  degree_spondylolisthesis   119.122881      0.0     0.279551         True

### 🔑 Question 2: Are any features linearly dependent on others (e.g., sacral slope and pelvic incidence)?
Highly Correlated Pairs (|r| > 0.7):
-----------------------------------
pelvic_incidence <-> lumbar_lordosis_angle: r = 0.717
pelvic_incidence <-> sacral_slope: r = 0.815

### 🔑 Question 3: Do biomechanical measurements cluster differently for normal vs. abnormal cases?
- There is no pure cluster separation

### 🔑 Question 4: Are there multicollinearity issues that impact modeling? Yes
⚠️  Features with high VIF (>10) indicating multicollinearity:
                 Feature        VIF
0       pelvic_incidence        inf
1            pelvic_tilt        inf
2  lumbar_lordosis_angle  18.942994
3           sacral_slope        inf
4          pelvic_radius  12.282573


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

✏️ class: has values like 'Normal', 'Hernia', 'Spondylolisthesis' → Low cardinality

---

#### 🔑 Question 1:
**Which biomechanical features are likely to have the most predictive power for classifying spinal conditions, and how did you determine that?**

💡 **Hint:**  
Use `.groupby(target).mean()`, boxplots, or violin plots to see how each feature separates across classes.  
Use correlation matrices or feature importance from an early tree-based model as a sanity check.  
Look for features that consistently differ between classes (e.g., high pelvic tilt in abnormal cases).

✏️ | Feature                       | Predictive Power (Visual Justification)                                                   |
| ----------------------------- | ----------------------------------------------------------------------------------------- |
| **degree\_spondylolisthesis** | Very strong separation, especially for `Spondylolisthesis`, which has much higher values. |
| **pelvic\_incidence**         | Clear difference, especially elevated in `Spondylolisthesis`.                             |
| **pelvic\_tilt**              | Distinct distributions; higher in `Spondylolisthesis`, moderate in `Hernia`.              |
| **sacral\_slope**             | Moderate separation; some overlap but still useful.                                       |
| **lumbar\_lordosis\_angle**   | Varies across classes; some predictive value but more overlap.                            |
| **pelvic\_radius**            | Least predictive; overlapping distributions among all classes.                            |

According the boxplots and violin plot

---

#### 🔑 Question 2:
**What numerical features in the dataset needed to be scaled before being input into a neural network, and what scaling method did you choose?**

💡 **Hint:**  
Use `df.describe()` and histograms to evaluate spread and skew.  
Neural networks are sensitive to feature scale.  
Choose between `StandardScaler`, `MinMaxScaler`, or log-transform based on distribution and range.

✏️ | Scenario                              | Recommended Scaler        |
| ------------------------------------- | ------------------------- |
| Values are roughly Gaussian           | `StandardScaler`          |
| Values are bounded (e.g., 0 to 1)     | `MinMaxScaler`            |
| Highly skewed or exponential features | Log-transform, then scale |

- Distribution Analysis:
-------------------------
                                           Feature  Skewness  Kurtosis  \
pelvic_incidence                  pelvic_incidence     0.520     0.224   
pelvic_tilt                            pelvic_tilt     0.677     0.676   
lumbar_lordosis_angle        lumbar_lordosis_angle     0.599     0.162   
sacral_slope                          sacral_slope     0.793     3.007   
pelvic_radius                        pelvic_radius    -0.177     0.935   
degree_spondylolisthesis  degree_spondylolisthesis     4.318    38.069   

                            Range     IQR  
pelvic_incidence          103.686  26.447  
pelvic_tilt                55.987  11.453  
lumbar_lordosis_angle     111.742  26.000  
sacral_slope              108.063  19.349  
pelvic_radius              92.988  14.758  
degree_spondylolisthesis  429.601  39.684  

Normality Tests (Shapiro-Wilk p-values):
-----------------------------------
pelvic_incidence         : p=0.000007 (Non-normal)
pelvic_tilt              : p=0.000001 (Non-normal)
lumbar_lordosis_angle    : p=0.000009 (Non-normal)
sacral_slope             : p=0.000001 (Non-normal)
pelvic_radius            : p=0.016610 (Non-normal)
degree_spondylolisthesis : p=0.000000 (Non-normal)

Degree of Spondylolisthesis has the highest skewness - > log transform, the rest StandardScaler()


---

#### 🔑 Question 3:
**Did you create any new features based on domain knowledge or feature interactions? If yes, what are they and why might they help the model better predict spinal conditions?**

💡 **Hint:**  
Try combining related anatomical angles or calculating ratios/differences (e.g., `pelvic_incidence - sacral_slope`).  
Think: which combinations might reflect spinal misalignment patterns?

✏️ PI/SS 
- Pelvic Incidence (PI) represents an anatomical constant — it doesn't change with posture.

Sacral Slope (SS) is posture-dependent, and varies with spinal curvature and pelvic tilt.

The PI/SS ratio captures the relationship between structure (PI) and posture (SS).

🔍 Why This Might Be Useful
A high PI/SS ratio could indicate that the sacrum is more horizontal (i.e., flatter slope), possibly compensating for spinal misalignment.

A low ratio may suggest that pelvic incidence and sacral slope are well-aligned (normal biomechanics).


---

#### 🔑 Question 4:
**Which features, if any, did you choose to drop or simplify before modeling, and what was your justification?**

💡 **Hint:**  
Check for highly correlated or constant features using `.corr()` and `.nunique()`.  
Avoid overfitting by removing redundant signals.  
Be cautious about leaking target-related info if any engineered features are overly specific.

✏️ **Answer:**  
- `pelvic_incidence` is strongly linearly dependent on `sacral_slope` and `pelvic_tilt`.  
  - Approximation: `pelvic_incidence ≈ sacral_slope + pelvic_tilt`.  
  - Including all three features can cause redundancy and multicollinearity, which is harmful in linear models and adds noise in deep learning.  
- To address this, `pelvic_incidence` was dropped, and the `pi_ss_ratio` feature was created to capture the relationship between `pelvic_incidence` and `sacral_slope`.

---

#### 🔑 Question 5:
**After preprocessing, what does your final input schema look like (i.e., how many numerical and categorical features)? Are there class imbalance or sparsity issues to be aware of?**

💡 **Hint:**  
Use `.shape`, `.dtypes`, and `.value_counts()` on the target.  
Check if certain features have many near-zero or rare values.  
Look for imbalanced class distributions and think about resampling, class weights, or focal loss.

✏️ **Answer:**  
- **Final Input Schema:**  
  - Numerical Features:  
    - `pelvic_tilt`  
    - `sacral_slope`  
    - `lumbar_lordosis_angle`  
    - `pelvic_radius`  
    - `degree_spondylolisthesis`  
    - `pi_ss_ratio`  
  - Categorical Feature:  
    - `class/binary_class` (target)

- **Class Imbalance:**  
  - **Class Distribution:**  
    - Spondylolisthesis: 150 (48.4%)  
    - Normal: 100 (32.3%)  
    - Hernia: 60 (19.4%)  
  - **Imbalance Ratio:** 2.50  

- **Analysis:**  
  - The dataset shows significant class imbalance, with the "Hernia" class being underrepresented.  
  - To address this, techniques such as resampling, class weights, or focal loss can be applied during model training.

---


---

### 📆 Week 2: Model Development & Experimentation

### 🔑 Question 1:

### 🔑 Question 2:

### 🔑 Question 3:

### 🔑 Question 4:

### 🔑 Question 5:

---

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
