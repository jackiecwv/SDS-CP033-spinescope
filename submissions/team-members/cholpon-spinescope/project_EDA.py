import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os
import warnings
warnings.filterwarnings("ignore")


################ Loading the dataset ################

path = kagglehub.dataset_download("uciml/biomechanical-features-of-orthopedic-patients")

print(f"Dataset downloaded to: {path}")
os.listdir(path)
data1 = pd.read_csv(os.path.join(path, "column_2C_weka.csv"))
data2 = pd.read_csv(os.path.join(path, "column_3C_weka.csv"))
pd.set_option('display.max_columns', None)  # Show all columns in the DataFrame

################# EDA on the first dataset #################


with open("reports.txt", "w") as f:
    f.write("Dataset 1: biomechanical-features-of-orthopedic-patients\n")
    f.write("--" * 40 + "\n")
    f.write(f"Shape: {data1.shape}\n")
    f.write("--" * 40 + "\n")
    f.write(f"Info:\n{data1.info()}\n")
    f.write("--" * 40 + "\n")
    f.write(f"Description:\n{data1.describe()}\n")
    f.write("--" * 40 + "\n")
    f.write(f"Missing values:\n{data1.isnull().sum()}\n")
    f.write("--" * 40 + "\n")
    f.write(f"Unique values:\n{data1.nunique()}\n")
    f.write("--" * 40 + "\n")
    f.write(f"Columns:\n{data1.columns.tolist()}\n")
    f.write("--" * 40 + "\n")
    f.write(f"Duplicated rows: {data1.duplicated().sum()}\n")
    f.write("--" * 40 + "\n")


# Visualizing the distribution of the target variable
