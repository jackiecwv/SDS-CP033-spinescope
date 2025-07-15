import pandas as pd
import numpy as np
import os
import joblib
import argparse
import mlflow
import wandb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

### HELPER FUNCTIONS ###
wandb.init(project="credit-card-fraud-detection-test-515", name="data-cleaning")

def load_data():
    """
    Load the raw dataset from WandB Artifacts.
    
    :return: pandas DataFrame
    """
    # Download the artifact from WandB
    artifact = wandb.use_artifact('anicin/credit-card-fraud-detection-test-515/original_data:latest', type='dataset')
    artifact_dir = artifact.download()

    # Load the dataset
    data = pd.read_csv(os.path.join(artifact_dir, 'balanced_dataset.csv'))
    wandb.log({"raw_data_shape": data.shape})
    
    return data

def save_data(data):
    """
    Save the cleaned dataset to WandB as an artifact.
    
    :param data: pandas DataFrame, cleaned data
    """
    # Save the cleaned data locally
    cleaned_data_path = 'processed_data.csv'
    data.to_csv(cleaned_data_path, index=False)
    
    # Log the cleaned data as an artifact in WandB
    artifact = wandb.Artifact(name='processed_data.csv', type='dataset')
    artifact.add_file(cleaned_data_path)
    wandb.log_artifact(artifact)
    wandb.log({"cleaned_data_shape": data.shape})

### DO NOT CHANGE THE CODE ABOVE THIS LINE ###

def clean_data(data):
    """
    Clean the dataset by handling missing values, outliers, and other anomalies.
    
    :param data: pandas DataFrame, raw data
    :return: pandas DataFrame, cleaned data
    """
    # Step 1: Handle Missing Values
    imputer = SimpleImputer(strategy="median")
    numeric_data = data.select_dtypes(include=[np.number])
    imputed_data = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)

    # TODO: Save imputer

    # Step 2: Handle Outliers
    outlier_threshold = 1.5
    for column in imputed_data.columns: 
        if column != "is_fraud": 
            Q1 = imputed_data[column].quantile(0.25) 
            Q3 = imputed_data[column].quantile(0.75) 
            IQR = Q3 - Q1 
            outlier_range = outlier_threshold * IQR 
            imputed_data = imputed_data[ 
                (imputed_data[column] >= Q1 - outlier_range) &  
                (imputed_data[column] <= Q3 + outlier_range) 
            ]

    # Log outlier removal stats
    outliers_removed = imputed_data.shape[0] - imputed_data.shape[0]
    
    # TODO: Log outlier removal stats

    # Step 3: Feature Scaling
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(imputed_data), columns=imputed_data.columns)

    # TODO: Log the feature scaling stats

    # TODO: Save scaler
)

    # Add trans_date_trans_time column back to the data 
    scaled_data = pd.concat([scaled_data, data['trans_date_trans_time']], axis=1) 
    scaled_data['is_fraud'] = data['is_fraud']

    return scaled_data


def main():
    mlflow.start_run()
    
    # Load data from WandB
    data = load_data()
    
    cleaned_data = clean_data(data)
    
    # Save cleaned data to WandB
    save_data(cleaned_data)
    
    # Log final parameters to MLFlow
    mlflow.log_param("cleaned_data_shape", cleaned_data.shape)

    mlflow.end_run()

if __name__ == "__main__":
    main()
