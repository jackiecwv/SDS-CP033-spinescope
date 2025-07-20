import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# Replace 'your_dataset.csv' with the actual dataset file
df = pd.read_csv('your_dataset.csv')

# Open a text file to write the outputs
output_file = open("analysis_report2_output.txt", "w")

# Question 1: High-cardinality categorical features
output_file.write("#### 🔑 Question 1: High-cardinality categorical features\n")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
output_file.write(f"Categorical columns: {list(categorical_cols)}\n\n")

for col in categorical_cols:
    unique_count = df[col].nunique()
    output_file.write(f"Feature: {col}, Unique Values: {unique_count}\n")
    if unique_count > 10:  # Assuming high-cardinality is >10 unique values
        output_file.write(f"⚠️ {col} is high-cardinality.\n")
        # Example encoding
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        output_file.write(f"{col} has been encoded using LabelEncoder.\n")
output_file.write("\n")

# Question 2: Biomechanical features with predictive power
output_file.write("#### 🔑 Question 2: Biomechanical features with predictive power\n")
target_col = 'target'  # Replace with the actual target column name
numerical_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col)

grouped_means = df.groupby(target_col).mean()
output_file.write("Mean values of numerical features grouped by target:\n")
output_file.write(grouped_means.to_string() + "\n\n")

# Correlation matrix
correlation_matrix = df[numerical_cols].corr()
output_file.write("Correlation matrix:\n")
output_file.write(correlation_matrix.to_string() + "\n\n")

# Feature importance (example using correlation with target)
correlation_with_target = df.corr()[target_col].sort_values(ascending=False)
output_file.write("Correlation of features with target:\n")
output_file.write(correlation_with_target.to_string() + "\n\n")

# Question 3: Scaling numerical features
output_file.write("#### 🔑 Question 3: Scaling numerical features\n")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[numerical_cols])
output_file.write("Numerical features have been scaled using StandardScaler.\n\n")

# Question 4: New features based on domain knowledge
output_file.write("#### 🔑 Question 4: New features based on domain knowledge\n")
df['pi_ss_ratio'] = df['pelvic_incidence'] / df['sacral_slope']
output_file.write("New feature 'pi_ss_ratio' created as pelvic_incidence / sacral_slope.\n\n")

# Question 5: Dropped or simplified features
output_file.write("#### 🔑 Question 5: Dropped or simplified features\n")
df = df.drop(columns=['pelvic_incidence'])  # Example of dropping a redundant feature
output_file.write("Dropped 'pelvic_incidence' due to high correlation with 'sacral_slope'.\n\n")

# Question 6: Final input schema
output_file.write("#### 🔑 Question 6: Final input schema\n")
output_file.write(f"Shape of the dataset: {df.shape}\n")
output_file.write(f"Numerical features: {len(numerical_cols)}\n")
output_file.write(f"Categorical features: {len(categorical_cols)}\n\n")

# Class imbalance
class_counts = df[target_col].value_counts()
output_file.write("Class distribution:\n")
output_file.write(class_counts.to_string() + "\n")
if class_counts.min() / class_counts.max() < 0.5:
    output_file.write("⚠️ Class imbalance detected. Consider resampling or using class weights.\n")

# Close the output file
output_file.close()

print("Analysis complete. All outputs have been written to 'analysis_report2_output.txt'.")