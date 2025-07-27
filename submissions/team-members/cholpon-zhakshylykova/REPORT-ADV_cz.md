📄 SpineScope – Project Report - 🔴 Advanced Track
Welcome to your personal project report!
This report answers key reflection questions for each phase of the project. It is designed to help you think like a data scientist, guide AI tools more effectively, and prepare for real-world job interviews.

✅ Phase 1: Setup & Exploratory Data Analysis (EDA)
Answer the EDA questions provided in the project materials here. Focus on data quality, trends, anomalies, and relationships.

🔑 Question 1: Which features are most strongly correlated with spinal abnormalities?
Feature Interactions Analysis
The following features are most strongly correlated with spinal abnormalities based on statistical analysis:

Feature	F-Statistic	P-Value	Effect Size	Significant
pelvic_incidence	98.54	0.0	0.243	✅ True
pelvic_tilt	21.30	0.0	0.065	✅ True
lumbar_lordosis_angle	114.98	0.0	0.272	✅ True
sacral_slope	89.64	0.0	0.226	✅ True
pelvic_radius	16.87	0.0	0.052	✅ True
degree_spondylolisthesis	119.12	0.0	0.280	✅ True
🔑 Question 2: Are any features linearly dependent on others (e.g., sacral slope and pelvic incidence)?
Highly Correlated Pairs (|r| > 0.7):
The following feature pairs exhibit strong linear dependence:

Feature Pair	Correlation (r)
pelvic_incidence ↔ lumbar_lordosis_angle	0.717
pelvic_incidence ↔ sacral_slope	0.815
🔑 Question 3: Do biomechanical measurements cluster differently for normal vs. abnormal cases?
Observation: There is no pure cluster separation between normal and abnormal cases.
PCA Analysis: Partial overlap exists between clusters.
t-SNE Visualization: Clusters are not fully distinct but show some separability.
🔑 Question 4: Are there multicollinearity issues that impact modeling?
⚠️ Features with High VIF (>10) Indicating Multicollinearity:
The following features exhibit multicollinearity:

Feature	VIF
pelvic_incidence	∞
pelvic_tilt	∞
lumbar_lordosis_angle	18.94
sacral_slope	∞
pelvic_radius	12.28
✅ Phase 2: Model Development
This phase spans 3 weeks. Answer each set of questions weekly as you build, train, evaluate, and improve your models.

📆 Week 1: Feature Engineering & Data Preprocessing
🔑 Question 1: Which categorical features are high-cardinality, and how will you encode them for use with embedding layers?
Answer:

Feature: class
Values: 'Normal', 'Hernia', 'Spondylolisthesis'
Cardinality: Low (3 unique values).
Encoding: One-hot encoding or integer mapping (e.g., LabelEncoder) is sufficient. Embedding layers are not required.
🔑 Question 2: Which biomechanical features are likely to have the most predictive power for classifying spinal conditions, and how did you determine that?
Answer:
The following features are most predictive based on boxplots, violin plots, and feature importance analysis:

Feature	Predictive Power (Visual Justification)
degree_spondylolisthesis	Very strong separation, especially for Spondylolisthesis, which has much higher values.
pelvic_incidence	Clear difference, especially elevated in Spondylolisthesis.
pelvic_tilt	Distinct distributions; higher in Spondylolisthesis, moderate in Hernia.
sacral_slope	Moderate separation; some overlap but still useful.
lumbar_lordosis_angle	Varies across classes; some predictive value but more overlap.
pelvic_radius	Least predictive; overlapping distributions among all classes.
🔑 Question 3: What numerical features in the dataset needed to be scaled before being input into a neural network, and what scaling method did you choose?
Answer:

Scenario	Recommended Scaler
Values are roughly Gaussian	StandardScaler
Values are bounded (e.g., 0 to 1)	MinMaxScaler
Highly skewed or exponential features	Log-transform, then scale
Distribution Analysis:

Degree of Spondylolisthesis has the highest skewness and kurtosis.
Action Taken: Log-transform this feature.
All other features were scaled using StandardScaler.
🔑 Question 4: Did you create any new features based on domain knowledge or feature interactions? If yes, what are they and why might they help the model better predict spinal conditions?
Answer:

Feature Created: pi_ss_ratio
Formula: pelvic_incidence / sacral_slope
Purpose: Captures the relationship between structure (pelvic incidence) and posture (sacral slope).
Why Useful:
A high ratio may indicate compensatory mechanisms for spinal misalignment.
A low ratio suggests normal biomechanics.
🔑 Question 5: Which features, if any, did you choose to drop or simplify before modeling, and what was your justification?
Answer:

Dropped Feature: pelvic_incidence
Reason: Strongly correlated with sacral_slope and pelvic_tilt.
Action Taken: Replaced with pi_ss_ratio to reduce redundancy and multicollinearity.
🔑 Question 6: After preprocessing, what does your final input schema look like (i.e., how many numerical and categorical features)? Are there class imbalance or sparsity issues to be aware of?
Answer:

Final Input Schema:

Numerical Features:
pelvic_tilt
sacral_slope
lumbar_lordosis_angle
pelvic_radius
degree_spondylolisthesis
pi_ss_ratio
Categorical Feature:
class/binary_class (target variable)
Class Imbalance:

Class Distribution:
Spondylolisthesis: 150 (48.4%)
Normal: 100 (32.3%)
Hernia: 60 (19.4%)
Imbalance Ratio: 2.50
Proposed Solutions:

Resampling Methods:
Oversampling (e.g., SMOTE) or undersampling.
Class Weights:
Assign higher weights to the minority class during training.
Loss Function Adjustments:
Use Focal Loss to focus on hard-to-classify samples.
📆 Week 2: Model Development & Experimentation
📆 Week 3: Model Tuning
✅ Phase 3: Model Deployment
