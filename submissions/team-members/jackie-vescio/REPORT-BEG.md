
### Week 1: Feature Engineering & Data Preprocessing

---

#### Question 1:
**Which biomechanical features show the strongest relationship with the target (spinal condition), and how did you determine this?**

The features showing the strongest relationship with the target (spinal condition) are `degree_spondylolisthesis`, `pelvic_incidence`, and `lumbar_lordosis_angle`. This was determined using `.groupby('class').mean()`, correlation heatmaps, and boxplots. These features showed clear separation between Normal and Abnormal cases, with `degree_spondylolisthesis` especially high in Abnormal cases.

---

#### Question 2:
**Before building any model, what patterns or class imbalances did you observe in the target variable? Will this affect your modeling choices?**

There is a slight class imbalance, with more Abnormal than Normal cases. This was checked using `.value_counts()` and bar plots. While not extreme, I will apply stratified train-test splits and consider class weights in some models to avoid bias toward the majority class.

---

#### Question 3:
**Which features appear skewed or contain outliers, and what transformations (if any) did you apply to address them?**

Features like `degree_spondylolisthesis`, `pelvic_tilt`, and `pelvic_incidence` showed noticeable skewness and outliers. I applied capping using the IQR method to reduce extreme values. For the engineered feature `sacral_slope_divided_by_pelvic_incidence`, I applied an additional 99th percentile cap after observing a long right tail with extreme outliers (max ~2034 reduced to ~8.6). These steps help prepare the data for models sensitive to extreme values.

---

#### Question 4:
**What scaling method did you apply to your numerical features, and why was it necessary (or not) for the algorithms you plan to use?**

I applied `StandardScaler` to standardize numerical features to zero mean and unit variance, preparing the data for algorithms like logistic regression, SVM, and KNN, which are sensitive to feature scales. For tree-based models, scaling is not required, but having standardized features keeps modeling flexible.

---

#### Question 5:
**Did you create any new features that might help distinguish between different spinal conditions? If yes, what are they and what was the reasoning behind them?**

Yes, I created two derived features:
- `pelvic_incidence_minus_pelvic_tilt` (difference between pelvic incidence and pelvic tilt)
- `sacral_slope_divided_by_pelvic_incidence` (ratio of sacral slope to pelvic incidence, with +1e-6 to avoid division by zero)

These features aim to capture biomechanical relationships not fully expressed by individual variables and may enhance the model’s ability to distinguish between normal and abnormal spinal conditions.

---

#### Additional Notes 

## Handling Skew and Scaling

During preprocessing, I considered applying a log transformation (e.g., `np.log1p()`) to reduce skew in highly right-skewed features, such as `degree_spondylolisthesis`. This can help normalize distributions and improve model performance, especially for models sensitive to non-normality.

I also considered using `RobustScaler` to rescale features using the median and interquartile range (IQR), which can be more robust to outliers compared to `StandardScaler`.

However, after capping extreme values using the IQR method, I decided that applying `StandardScaler` would be sufficient for preparing the data, especially since I plan to experiment with both scale-sensitive (e.g., logistic regression, SVM) and scale-insensitive (e.g., tree-based) models

---


