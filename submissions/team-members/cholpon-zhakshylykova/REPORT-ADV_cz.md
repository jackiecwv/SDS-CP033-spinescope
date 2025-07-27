# 📄 SpineScope – Project Report  
**Track:** 🔴 Advanced | **Role:** Aspiring Data Scientist  

Welcome to your personal project report! This document reflects your end-to-end process through each phase of the SpineScope challenge. It’s structured to build your problem-solving mindset, improve how you guide AI tools, and prepare you for real-world data science interviews.

---

## ✅ Phase 1: Setup & Exploratory Data Analysis (EDA)

This phase focuses on understanding the data, evaluating quality, and identifying patterns or issues relevant to spinal abnormalities.

### 🔑 Q1: Which features are most strongly correlated with spinal abnormalities?

| Feature                    | F-Statistic | P-Value | Effect Size | Significant |
|----------------------------|-------------|---------|-------------|-------------|
| `pelvic_incidence`         | 98.54       | 0.0     | 0.243       | ✅ Yes       |
| `pelvic_tilt`              | 21.30       | 0.0     | 0.065       | ✅ Yes       |
| `lumbar_lordosis_angle`    | 114.98      | 0.0     | 0.272       | ✅ Yes       |
| `sacral_slope`             | 89.64       | 0.0     | 0.226       | ✅ Yes       |
| `pelvic_radius`            | 16.87       | 0.0     | 0.052       | ✅ Yes       |
| `degree_spondylolisthesis` | 119.12      | 0.0     | 0.280       | ✅ Yes       |

---

### 🔑 Q2: Are any features linearly dependent on others?

**Highly Correlated Pairs** (|r| > 0.7):

| Feature Pair                               | Correlation (r) |
|--------------------------------------------|------------------|
| `pelvic_incidence` ↔ `lumbar_lordosis_angle` | 0.717            |
| `pelvic_incidence` ↔ `sacral_slope`          | 0.815            |

---

### 🔑 Q3: Do biomechanical features cluster by class?

- **PCA:** Some cluster separability, but partial overlap.  
- **t-SNE:** Slight class-specific patterns, but not fully distinct.  
- **Conclusion:** No pure clustering; abnormal cases intermingle with normal.

---

### 🔑 Q4: Are there multicollinearity issues?

**Features with High VIF (>10):**

| Feature                 | VIF     |
|------------------------|---------|
| `pelvic_incidence`     | ∞       |
| `pelvic_tilt`          | ∞       |
| `lumbar_lordosis_angle`| 18.94   |
| `sacral_slope`         | ∞       |
| `pelvic_radius`        | 12.28   |

🛠️ **Action Taken:** Dropped or reengineered features (see Phase 2) to mitigate multicollinearity.

---

## ✅ Phase 2: Model Development

### 📆 Week 1: Feature Engineering & Preprocessing

---

### 🔑 Q1: How did you handle categorical features?

- **Feature:** `class`  
- **Cardinality:** Low (3 values: Normal, Hernia, Spondylolisthesis)  
- **Encoding:** One-hot or Label Encoding. No embeddings needed.

---

### 🔑 Q2: Most predictive biomechanical features?

| Feature                    | Predictive Power (Visual Justification)              |
|----------------------------|-----------------------------------------------------|
| `degree_spondylolisthesis` | Strong separation for Spondylolisthesis             |
| `pelvic_incidence`         | Elevated in Spondylolisthesis                       |
| `pelvic_tilt`              | Higher in Spondylolisthesis, moderate in Hernia     |
| `sacral_slope`             | Moderate separation                                 |
| `lumbar_lordosis_angle`    | Some overlap, still useful                          |
| `pelvic_radius`            | Least predictive, heavily overlapping               |

---

### 🔑 Q3: Scaling strategy for numerical features?

| Scenario                                  | Scaler Used      |
|-------------------------------------------|------------------|
| Roughly Gaussian                          | `StandardScaler` |
| Bounded (e.g., [0, 1])                    | `MinMaxScaler`   |
| Highly skewed or exponential              | Log-transform    |

- `degree_spondylolisthesis`: Log-transformed due to high skewness.  
- All other features: Standardized.

---

### 🔑 Q4: Were any new features created?

✅ **Yes**  
- **Feature:** `pi_ss_ratio`  
- **Formula:** `pelvic_incidence / sacral_slope`  
- **Rationale:** Captures structural-to-postural relationship. A high value may indicate spinal compensation mechanisms.

---

### 🔑 Q5: Did you drop or simplify features?

- **Dropped:** `pelvic_incidence`  
- **Reason:** High multicollinearity with other features.  
- **Replacement:** Used `pi_ss_ratio` to preserve meaningful info with reduced redundancy.

---

### 🔑 Q6: Final input schema & class balance?

**Input Features:**

- **Numerical (6):**  
  `pelvic_tilt`, `sacral_slope`, `lumbar_lordosis_angle`,  
  `pelvic_radius`, `degree_spondylolisthesis`, `pi_ss_ratio`

- **Target:** `class` or `binary_class`

**Class Imbalance:**

| Class             | Count | %     |
|-------------------|-------|-------|
| Spondylolisthesis | 150   | 48.4% |
| Normal            | 100   | 32.3% |
| Hernia            | 60    | 19.4% |

- **Imbalance Ratio:** 2.5  

**Mitigation Strategies:**
- Oversampling (e.g., SMOTE)  
- Class weighting  
- Loss function adjustment (e.g., Focal Loss)

---

### 📆 Week 2: Model Building & Experimentation

> *To be completed – document training progress, architecture choices, and evaluation metrics.*

---

### 📆 Week 3: Model Tuning & Finalization

> *To be completed – log hyperparameter tuning, cross-validation results, and final metrics.*

---

## ✅ Phase 3: Model Deployment

> *Placeholder for deployment strategy, model export, and integration plan.*
