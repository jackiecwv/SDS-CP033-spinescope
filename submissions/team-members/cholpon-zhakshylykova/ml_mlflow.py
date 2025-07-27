import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, 
    RandomizedSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import json
import pickle
from typing import Dict, List, Tuple, Any
import shap
from sklearn.model_selection import learning_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from sklearn.pipeline import Pipeline

# --- ENSURE OUTPUT DIRECTORY EXISTS ---
os.makedirs('outputs_ml', exist_ok=True)

# Configuration
class Config:
    """Configuration class for model development pipeline"""
    EXPERIMENT_NAME = "Orthopedic_Patients_Classification"
    MODEL_REGISTRY_NAME = "orthopedic_classifier"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2  # From training set
    CV_FOLDS = 5
    MAX_EVALS = 50  # For hyperparameter tuning
    
    # Class imbalance handling
    IMBALANCE_STRATEGY = "SMOTE"  # Options: "SMOTE", "UNDERSAMPLING", "SMOTEENN", "WEIGHTED"
    
    # Linear model preprocessing
    POWER_TRANSFORM = True  # Apply Yeo-Johnson transformation
    VIF_THRESHOLD = 5.0  # Variance Inflation Factor threshold
    OUTLIER_REMOVAL = True  # Remove outliers for linear models
    
    # MLflow tracking
    TRACKING_URI = "sqlite:///mlflow.db"  # Use SQLite for local tracking
    ARTIFACT_ROOT = "./mlruns"

class ModelDevelopmentPipeline:
    """Comprehensive model development pipeline with MLflow tracking"""
    
    def __init__(self, config: Config):
        self.config = config
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = None
        self.standard_scaler = None
        self.power_transformer = None
        self.label_encoder = None
        self.feature_names = None
        self.selected_features = None
        self.models_performance = {}
        self.smote = None
        self.outlier_mask = None
        
        # Setup MLflow
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Initialize MLflow tracking"""
        mlflow.set_tracking_uri(self.config.TRACKING_URI)
        
        # Create experiment if it doesn't exist
        try:
            experiment_id = mlflow.create_experiment(
                name=self.config.EXPERIMENT_NAME,
                artifact_location=self.config.ARTIFACT_ROOT
            )
        except mlflow.exceptions.MlflowException:
            experiment_id = mlflow.get_experiment_by_name(self.config.EXPERIMENT_NAME).experiment_id
        
        mlflow.set_experiment(self.config.EXPERIMENT_NAME)
        print(f"MLflow experiment: {self.config.EXPERIMENT_NAME}")
        print(f"Experiment ID: {experiment_id}")
    
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("Loading and preparing data...")
        

        self.df = pd.read_csv('column_3C_processed.csv')

        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Target distribution:\n{self.df['binary_class'].value_counts()}")
        
        # Log dataset info to MLflow
        with mlflow.start_run(run_name="data_preparation"):
            mlflow.log_param("dataset_shape", self.df.shape)
            mlflow.log_param("n_features", len(self.df.select_dtypes(include=[np.number]).columns))
            mlflow.log_param("target_classes", list(self.df['binary_class'].unique()))
            mlflow.log_param("class_distribution", dict(self.df['binary_class'].value_counts()))
            
            # Log data quality metrics
            mlflow.log_metric("missing_values", self.df.isnull().sum().sum())
            mlflow.log_metric("duplicate_rows", self.df.duplicated().sum())
            
            # Save dataset info
            dataset_info = {
                "shape": self.df.shape,
                "columns": list(self.df.columns),
                "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
                "missing_values": self.df.isnull().sum().to_dict(),
                "class_distribution": self.df['binary_class'].value_counts().to_dict()
            }
            
            # --- OUTPUTS TO outputs_ml FOLDER ---
            with open("outputs_ml/dataset_info.json", "w") as f:
                json.dump(dataset_info, f, indent=2)
            mlflow.log_artifact("outputs_ml/dataset_info.json")
            os.remove("outputs_ml/dataset_info.json")
    
    def check_linear_model_assumptions(self, X, y):
        print("Checking linear model assumptions...")

        assumptions_results = {}

        # 1. Check for multicollinearity using VIF
        print("  Checking multicollinearity...")
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(X.shape[1])]

        high_vif_features = vif_data[vif_data["VIF"] > self.config.VIF_THRESHOLD]["Feature"].tolist()
        assumptions_results["high_vif_features"] = high_vif_features
        assumptions_results["vif_data"] = vif_data

        print(f"    Features with VIF > {self.config.VIF_THRESHOLD}: {len(high_vif_features)}")

        # 2. Check normality of features
        print("  Checking feature normality...")
        normality_results = {}
        for col in X.columns:
            stat, p_value = stats.shapiro(X[col])
            # Convert numpy types to Python types for JSON
            normality_results[col] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_normal": bool(p_value > 0.05)
            }

        non_normal_features = [col for col, result in normality_results.items() 
                             if not result["is_normal"]]
        assumptions_results["non_normal_features"] = non_normal_features
        assumptions_results["normality_results"] = normality_results

        print(f"    Non-normal features: {len(non_normal_features)}")

        # 3. Check for outliers using IQR method
        print("  Checking for outliers...")
        outlier_counts = {}
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
            outlier_counts[col] = int(outliers)

        assumptions_results["outlier_counts"] = outlier_counts
        total_outliers = sum(outlier_counts.values())
        print(f"    Total outliers detected: {total_outliers}")

        # --- OUTPUTS TO outputs_ml FOLDER ---
        vif_data_path = "outputs_ml/vif_data.csv"
        vif_data.to_csv(vif_data_path, index=False)
        normality_results_path = "outputs_ml/normality_results.json"
        with open(normality_results_path, "w") as f:
            json.dump(normality_results, f, indent=2)
        outlier_counts_path = "outputs_ml/outlier_counts.json"
        with open(outlier_counts_path, "w") as f:
            json.dump(outlier_counts, f, indent=2)

        return assumptions_results   
     
    def apply_transformations(self, X_train, X_val, X_test, for_linear_models=False):
        """Apply transformations to meet linear model assumptions"""
        print("Applying data transformations...")

        X_train_transformed = X_train.copy()
        X_val_transformed = X_val.copy()
        X_test_transformed = X_test.copy()

        # Apply power transformation to degree_spondylolisthesis if present
        if 'degree_spondylolisthesis' in X_train_transformed.columns:
            print("  Applying power transformation to degree_spondylolisthesis...")

            # Use Yeo-Johnson which handles negative values, zero, and positive values
            self.degree_power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)

            # Extract the column for transformation
            train_col = X_train_transformed[['degree_spondylolisthesis']]
            val_col = X_val_transformed[['degree_spondylolisthesis']]
            test_col = X_test_transformed[['degree_spondylolisthesis']]

            # Fit on training data and transform all sets
            transformed_train = self.degree_power_transformer.fit_transform(train_col)
            transformed_val = self.degree_power_transformer.transform(val_col)
            transformed_test = self.degree_power_transformer.transform(test_col)

            # Update the DataFrames
            X_train_transformed['degree_spondylolisthesis'] = transformed_train.flatten()
            X_val_transformed['degree_spondylolisthesis'] = transformed_val.flatten()
            X_test_transformed['degree_spondylolisthesis'] = transformed_test.flatten()

            print(f"    Power transformation applied (lambda: {self.degree_power_transformer.lambdas_[0]:.4f})")
        else:
            print("  Warning: degree_spondylolisthesis column not found in dataset.")

        if for_linear_models:
            # Apply standard scaling to ALL columns (including degree_spondylolisthesis)
            print("  Applying standard scaling to all columns...")
            self.standard_scaler = StandardScaler()
            X_train_transformed = pd.DataFrame(
                self.standard_scaler.fit_transform(X_train_transformed),
                columns=X_train_transformed.columns,
                index=X_train_transformed.index
            )
            X_val_transformed = pd.DataFrame(
                self.standard_scaler.transform(X_val_transformed),
                columns=X_val_transformed.columns,
                index=X_val_transformed.index
            )
            X_test_transformed = pd.DataFrame(
                self.standard_scaler.transform(X_test_transformed),
                columns=X_test_transformed.columns,
                index=X_test_transformed.index
            )

            # Feature selection to address multicollinearity
            print("  Applying feature selection...")
            # Use SelectKBest with f_classif to select most informative features
            k_best = min(len(X_train_transformed.columns) - 1, 4)  # Select top 4 features
            selector = SelectKBest(score_func=f_classif, k=k_best)
            X_train_transformed = selector.fit_transform(X_train_transformed, self.y_train)
            X_val_transformed = selector.transform(X_val_transformed)
            X_test_transformed = selector.transform(X_test_transformed)

            # Update feature names
            self.selected_features = [self.feature_names[i] for i in selector.get_support(indices=True)]
            print(f"    Selected features: {self.selected_features}")

        else:
            # Standard scaling for non-linear models (ALL columns)
            print("  Applying standard scaling to all columns...")
            self.scaler = StandardScaler()
            X_train_transformed = self.scaler.fit_transform(X_train_transformed)
            X_val_transformed = self.scaler.transform(X_val_transformed)
            X_test_transformed = self.scaler.transform(X_test_transformed)

        return X_train_transformed, X_val_transformed, X_test_transformed
     
    def handle_class_imbalance(self, X_train, y_train):
        """Handle class imbalance using various techniques"""
        print(f"Handling class imbalance using {self.config.IMBALANCE_STRATEGY}...")
        
        original_distribution = pd.Series(y_train).value_counts().sort_index()
        print(f"  Original distribution: {dict(original_distribution)}")
        
        X_resampled, y_resampled = X_train, y_train
        
        if self.config.IMBALANCE_STRATEGY == "SMOTE":
            self.smote = SMOTE(random_state=self.config.RANDOM_STATE)
            X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
            
        elif self.config.IMBALANCE_STRATEGY == "UNDERSAMPLING":
            undersampler = RandomUnderSampler(random_state=self.config.RANDOM_STATE)
            X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
            
        elif self.config.IMBALANCE_STRATEGY == "SMOTEENN":
            smote_enn = SMOTEENN(random_state=self.config.RANDOM_STATE)
            X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
        
        # If using weighted approach, return original data
        elif self.config.IMBALANCE_STRATEGY == "WEIGHTED":
            print("  Using class weights in models...")
            return X_train, y_train
        
        new_distribution = pd.Series(y_resampled).value_counts().sort_index()
        print(f"  New distribution: {dict(new_distribution)}")
        
        return X_resampled, y_resampled
    
    def split_and_scale_data(self, target_col='binary_class'):
        """Split data into train/validation/test sets and scale features"""
        print("Splitting and scaling data...")
        
        # Prepare features and target
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        X = self.df[numerical_cols]
        y = self.df[target_col]
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.feature_names = numerical_cols
        
        # Split data: 60% train, 20% validation, 20% test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y_encoded, test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE, stratify=y_encoded
        )
        
        val_size_adj = self.config.VALIDATION_SIZE / (1 - self.config.TEST_SIZE)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj,
            random_state=self.config.RANDOM_STATE, stratify=y_temp
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Validation set: {self.X_val.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        # Check linear model assumptions
        assumptions = self.check_linear_model_assumptions(self.X_train, self.y_train)
        
        # Apply transformations for linear models
        self.X_train_linear, self.X_val_linear, self.X_test_linear = self.apply_transformations(
            self.X_train, self.X_val, self.X_test, for_linear_models=True
        )
        
        # Standard scaling for non-linear models
        self.X_train_scaled, self.X_val_scaled, self.X_test_scaled = self.apply_transformations(
            self.X_train, self.X_val, self.X_test, for_linear_models=False
        )
        
        # Handle class imbalance
        self.X_train_balanced, self.y_train_balanced = self.handle_class_imbalance(
            self.X_train_scaled, self.y_train
        )
        
        # For linear models, also balance the transformed data
        self.X_train_linear_balanced, self.y_train_linear_balanced = self.handle_class_imbalance(
            self.X_train_linear, self.y_train
        )
        
        # Log data split info
        with mlflow.start_run(run_name="data_splitting"):
            mlflow.log_param("train_size", len(self.X_train))
            mlflow.log_param("val_size", len(self.X_val))
            mlflow.log_param("test_size", len(self.X_test))
            mlflow.log_param("n_features", len(self.feature_names))
            mlflow.log_param("feature_names", self.feature_names)
            mlflow.log_param("target_encoding", dict(zip(self.label_encoder.classes_, 
                                                       self.label_encoder.transform(self.label_encoder.classes_))))
            mlflow.log_param("imbalance_strategy", self.config.IMBALANCE_STRATEGY)
            mlflow.log_param("power_transform", self.config.POWER_TRANSFORM)
            mlflow.log_param("vif_threshold", self.config.VIF_THRESHOLD)
            
            # Log assumption check results
            mlflow.log_param("high_vif_features", assumptions["high_vif_features"])
            mlflow.log_param("non_normal_features", assumptions["non_normal_features"])
            mlflow.log_metric("total_outliers", sum(assumptions["outlier_counts"].values()))
            mlflow.log_metric("balanced_train_size", len(self.X_train_balanced))
            
            if self.selected_features:
                mlflow.log_param("selected_features", self.selected_features)
    
    def get_model_configurations(self) -> Dict[str, Dict]:
        """Define model configurations with hyperparameter grids"""
        # Determine if we should use class weights
        use_class_weights = self.config.IMBALANCE_STRATEGY == "WEIGHTED"
        
        return {
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.config.RANDOM_STATE, max_iter=2000),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga'],
                    'class_weight': ['balanced'] if use_class_weights else [None, 'balanced']
                },
                'is_linear': True
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.config.RANDOM_STATE),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None],
                    'class_weight': ['balanced'] if use_class_weights else [None, 'balanced']
                },
                'is_linear': False
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.config.RANDOM_STATE),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'is_linear': False
            },
            'svm': {
                'model': SVC(random_state=self.config.RANDOM_STATE, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'poly', 'linear'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'class_weight': ['balanced'] if use_class_weights else [None, 'balanced']
                },
                'is_linear': True  # Can be linear with linear kernel
            },
            'naive_bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': np.logspace(-10, -6, 10)
                },
                'is_linear': True
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=self.config.RANDOM_STATE),
                'params': {
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'max_features': ['sqrt', 'log2', None],
                    'class_weight': ['balanced'] if use_class_weights else [None, 'balanced']
                },
                'is_linear': False
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                },
                'is_linear': False
            },
            'mlp': {
                'model': MLPClassifier(random_state=self.config.RANDOM_STATE, max_iter=2000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam', 'lbfgs'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                },
                'is_linear': False
            }
        }
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
        }
        
        # Add AUC if probabilities available
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
            metrics['avg_precision'] = average_precision_score(y_test, y_prob)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_test, y_test, cv=self.config.CV_FOLDS, scoring='accuracy')
        metrics['cv_accuracy_mean'] = cv_scores.mean()
        metrics['cv_accuracy_std'] = cv_scores.std()
        
        return metrics
    
    def create_evaluation_plots(self, model, X_test, y_test, model_name: str, feature_names: List[str]):
        """Create comprehensive evaluation plots"""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title(f'Confusion Matrix - {model_name}')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # ROC Curve
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = roc_auc_score(y_test, y_prob)
            axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            axes[0, 1].plot([0, 1], [0, 1], 'k--')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title(f'ROC Curve - {model_name}')
            axes[0, 1].legend()
        
        # Precision-Recall Curve
        if y_prob is not None:
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            avg_precision = average_precision_score(y_test, y_prob)
            axes[1, 0].plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title(f'Precision-Recall Curve - {model_name}')
            axes[1, 0].legend()
        
        # Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'])
            axes[1, 1].set_title(f'Feature Importance - {model_name}')
            axes[1, 1].set_xlabel('Importance')
        elif hasattr(model, 'coef_'):
            coef_importance = pd.DataFrame({
                'feature': feature_names,
                'coefficient': np.abs(model.coef_[0])
            }).sort_values('coefficient', ascending=True)
            
            axes[1, 1].barh(coef_importance['feature'], coef_importance['coefficient'])
            axes[1, 1].set_title(f'Feature Coefficients - {model_name}')
            axes[1, 1].set_xlabel('Absolute Coefficient')
        
        plt.tight_layout()
        out_path = f'outputs_ml/{model_name}_evaluation.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return out_path
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models with hyperparameter tuning"""
        print("Training and evaluating models...")
        
        model_configs = self.get_model_configurations()
        
        for model_name, config in model_configs.items():
            print(f"\n{'='*50}")
            print(f"Training {model_name.upper()}")
            print('='*50)
            
            with mlflow.start_run(run_name=f"{model_name}_training"):
                try:
                    # Log model configuration
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_param("random_state", self.config.RANDOM_STATE)
                    mlflow.log_param("is_linear_model", config.get('is_linear', False))
                    
                    # Determine which dataset to use based on model type
                    is_linear = config.get('is_linear', False)
                    
                    if is_linear:
                        # Use transformed data for linear models
                        if self.config.IMBALANCE_STRATEGY == "WEIGHTED":
                            X_train_use = self.X_train_linear
                            y_train_use = self.y_train
                        else:
                            X_train_use = self.X_train_linear_balanced
                            y_train_use = self.y_train_linear_balanced
                        
                        X_val_use = self.X_val_linear
                        X_test_use = self.X_test_linear
                        feature_names = self.selected_features if self.selected_features else self.feature_names
                        
                    else:
                        # Use standard scaled data for non-linear models
                        if self.config.IMBALANCE_STRATEGY == "WEIGHTED":
                            X_train_use = self.X_train_scaled
                            y_train_use = self.y_train
                        else:
                            X_train_use = self.X_train_balanced
                            y_train_use = self.y_train_balanced
                        
                        X_val_use = self.X_val_scaled
                        X_test_use = self.X_test_scaled
                        feature_names = self.feature_names
                    
                    mlflow.log_param("data_preprocessing", "linear_transformed" if is_linear else "standard_scaled")
                    mlflow.log_param("train_samples", len(X_train_use))
                    mlflow.log_param("features_used", feature_names)
                    
                    # Hyperparameter tuning
                    print("Performing hyperparameter tuning...")
                    
                    # Use RandomizedSearchCV for faster tuning
                    search = RandomizedSearchCV(
                        estimator=config['model'],
                        param_distributions=config['params'],
                        n_iter=min(self.config.MAX_EVALS, 
                                 np.prod([len(v) if isinstance(v, list) else 1 for v in config['params'].values()])),
                        cv=StratifiedKFold(n_splits=self.config.CV_FOLDS, shuffle=True, 
                                         random_state=self.config.RANDOM_STATE),
                        scoring='f1_weighted',
                        n_jobs=-1,
                        random_state=self.config.RANDOM_STATE,
                        verbose=1
                    )
                    
                    # Fit on training data
                    search.fit(X_train_use, y_train_use)
                    
                    # Best model
                    best_model = search.best_estimator_
                    
                    # Log best parameters
                    mlflow.log_params(search.best_params_)
                    mlflow.log_metric("best_cv_score", search.best_score_)
                    
                    # Evaluate on validation set
                    val_metrics = self.evaluate_model(best_model, X_val_use, self.y_val, model_name)
                    
                    # Log validation metrics
                    for metric_name, metric_value in val_metrics.items():
                        mlflow.log_metric(f"val_{metric_name}", metric_value)
                    
                    # Evaluate on test set
                    test_metrics = self.evaluate_model(best_model, X_test_use, self.y_test, model_name)
                    
                    # Log test metrics
                    for metric_name, metric_value in test_metrics.items():
                        mlflow.log_metric(f"test_{metric_name}", metric_value)
                    
                    # Store performance for comparison
                    self.models_performance[model_name] = {
                        'model': best_model,
                        'best_params': search.best_params_,
                        'val_metrics': val_metrics,
                        'test_metrics': test_metrics,
                        'is_linear': is_linear,
                        'X_test_use': X_test_use,
                        'feature_names': feature_names
                    }
                    
                    # Create evaluation plots
                    plot_path = self.create_evaluation_plots(best_model, X_test_use, self.y_test, model_name, feature_names)
                    mlflow.log_artifact(plot_path)
                    os.remove(plot_path)
                    
                    # Log model
                    signature = infer_signature(X_train_use, y_train_use)
                    mlflow.sklearn.log_model(
                        sk_model=best_model,
                        artifact_path=f"model_{model_name}",
                        signature=signature,
                        input_example=X_train_use[:5]
                    )
                    
                    # Feature importance analysis
                    if hasattr(best_model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'feature': feature_names,
                            'importance': best_model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        # Log top features
                        top_features = feature_importance.head(5)['feature'].tolist()
                        mlflow.log_param("top_5_features", top_features)
                        
                        # --- OUTPUTS TO outputs_ml FOLDER ---
                        fi_path = f'outputs_ml/{model_name}_feature_importance.csv'
                        feature_importance.to_csv(fi_path, index=False)
                        mlflow.log_artifact(fi_path)
                        os.remove(fi_path)
                    
                    print(f"✓ {model_name} training completed")
                    print(f"  Best validation F1: {val_metrics['f1_score']:.4f}")
                    print(f"  Test F1: {test_metrics['f1_score']:.4f}")
                    
                except Exception as e:
                    print(f"✗ Error training {model_name}: {str(e)}")
                    mlflow.log_param("error", str(e))
                    continue
    
    def compare_models(self):
        """Compare all trained models and select the best one"""
        print("\n" + "="*60)
        print("MODEL COMPARISON AND SELECTION")
        print("="*60)
        
        with mlflow.start_run(run_name="model_comparison"):
            # Create comparison dataframe
            comparison_data = []
            
            for model_name, performance in self.models_performance.items():
                row = {
                    'Model': model_name,
                    'Val_Accuracy': performance['val_metrics']['accuracy'],
                    'Val_Precision': performance['val_metrics']['precision'],
                    'Val_Recall': performance['val_metrics']['recall'],
                    'Val_F1': performance['val_metrics']['f1_score'],
                    'Test_Accuracy': performance['test_metrics']['accuracy'],
                    'Test_Precision': performance['test_metrics']['precision'],
                    'Test_Recall': performance['test_metrics']['recall'],
                    'Test_F1': performance['test_metrics']['f1_score'],
                }
                
                if 'roc_auc' in performance['test_metrics']:
                    row['Test_ROC_AUC'] = performance['test_metrics']['roc_auc']
                
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Test_F1', ascending=False)
            
            print("Model Performance Comparison:")
            print(comparison_df.round(4))
            
            # --- OUTPUTS TO outputs_ml FOLDER ---
            comp_csv_path = 'outputs_ml/model_comparison.csv'
            comparison_df.to_csv(comp_csv_path, index=False)
            mlflow.log_artifact(comp_csv_path)
            
            # Log best model info
            best_model_name = comparison_df.iloc[0]['Model']
            best_model_f1 = comparison_df.iloc[0]['Test_F1']
            
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_model_f1", best_model_f1)
            
            print(f"\n🏆 Best Model: {best_model_name} (Test F1: {best_model_f1:.4f})")
            
            # Create comparison visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Test metrics comparison
            metrics_to_plot = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1']
            
            for i, metric in enumerate(metrics_to_plot):
                ax = axes[i//2, i%2]
                bars = ax.bar(comparison_df['Model'], comparison_df[metric])
                ax.set_title(f'{metric.replace("_", " ")} Comparison')
                ax.set_ylabel(metric.replace("_", " "))
                ax.tick_params(axis='x', rotation=45)
                
                # Highlight best model
                best_idx = comparison_df[metric].idxmax()
                bars[best_idx].set_color('gold')
                
                # Add value labels
                for j, v in enumerate(comparison_df[metric]):
                    ax.text(j, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            comp_png_path = 'outputs_ml/model_comparison_chart.png'
            plt.savefig(comp_png_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(comp_png_path)
            plt.close()
            
            # Clean up
            os.remove(comp_csv_path)
            os.remove(comp_png_path)
            
            return best_model_name
    
    def advanced_model_analysis(self, best_model_name: str):
        """Perform advanced analysis on the best model"""
        print(f"\n" + "="*60)
        print(f"ADVANCED ANALYSIS - {best_model_name.upper()}")
        print("="*60)
        
        best_model_info = self.models_performance[best_model_name]
        best_model = best_model_info['model']
        is_linear = best_model_info['is_linear']
        X_test_use = best_model_info['X_test_use']
        feature_names = best_model_info['feature_names']
        
        with mlflow.start_run(run_name=f"{best_model_name}_advanced_analysis"):
            
            # 1. Learning Curves
            print("Generating learning curves...")
            train_sizes, train_scores, val_scores = learning_curve(
                best_model, X_test_use, self.y_test, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10)
            )
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training Score')
            plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation Score')
            plt.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                           train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
            plt.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                           val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
            plt.xlabel('Training Set Size')
            plt.ylabel('Score')
            plt.title(f'Learning Curves - {best_model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            lc_path = 'outputs_ml/learning_curves.png'
            plt.savefig(lc_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(lc_path)
            plt.close()
            
            # 2. Permutation Importance
            print("Calculating permutation importance...")
            perm_importance = permutation_importance(
                best_model, X_test_use, self.y_test, 
                n_repeats=10, random_state=self.config.RANDOM_STATE
            )
            
            perm_df = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            plt.figure(figsize=(10, 6))
            plt.barh(perm_df['feature'], perm_df['importance_mean'], 
                    xerr=perm_df['importance_std'])
            plt.xlabel('Permutation Importance')
            plt.title(f'Permutation Importance - {best_model_name}')
            plt.tight_layout()
            perm_path = 'outputs_ml/permutation_importance.png'
            plt.savefig(perm_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(perm_path)
            plt.close()
            
            # 3. Model Calibration
            print("Analyzing model calibration...")
            if hasattr(best_model, 'predict_proba'):
                y_prob = best_model.predict_proba(X_test_use)[:, 1]
                
                # Calibration plot
                from sklearn.calibration import calibration_curve
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    self.y_test, y_prob, n_bins=10
                )
                
                plt.figure(figsize=(10, 6))
                plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                        label=f"{best_model_name}")
                plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                plt.xlabel("Mean Predicted Probability")
                plt.ylabel("Fraction of Positives")
                plt.title(f'Calibration Plot - {best_model_name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                calib_path = 'outputs_ml/calibration_plot.png'
                plt.savefig(calib_path, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(calib_path)
                plt.close()
            
            # 4. SHAP values (if supported)
            try:
                print("Computing SHAP values...")
                explainer = shap.Explainer(best_model, X_test_use)
                shap_values = explainer(X_test_use)
                shap.summary_plot(shap_values, X_test_use, feature_names=feature_names, show=False)
                plt.tight_layout()
                shap_path = 'outputs_ml/shap_summary.png'
                plt.savefig(shap_path, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(shap_path)
                plt.close()
            except Exception as e:
                print(f"SHAP analysis skipped: {e}")
            
            # 5. Linear model assumptions validation (for linear models)
            if is_linear:
                print("Validating linear model assumptions...")
                try:
                    # Check linearity assumption with residual plots
                    y_pred = best_model.predict(X_test_use)
                    residuals = self.y_test - y_pred
                    
                    plt.figure(figsize=(12, 8))
                    
                    # Residuals vs Fitted
                    plt.subplot(2, 2, 1)
                    plt.scatter(y_pred, residuals, alpha=0.6)
                    plt.axhline(y=0, color='r', linestyle='--')
                    plt.xlabel('Fitted Values')
                    plt.ylabel('Residuals')
                    plt.title('Residuals vs Fitted Values')
                    
                    # Q-Q plot for normality of residuals
                    plt.subplot(2, 2, 2)
                    stats.probplot(residuals, dist="norm", plot=plt)
                    plt.title('Q-Q Plot of Residuals')
                    
                    # Histogram of residuals
                    plt.subplot(2, 2, 3)
                    plt.hist(residuals, bins=20, density=True, alpha=0.7)
                    plt.xlabel('Residuals')
                    plt.ylabel('Density')
                    plt.title('Distribution of Residuals')
                    
                    # Scale-Location plot
                    plt.subplot(2, 2, 4)
                    standardized_residuals = np.sqrt(np.abs(residuals))
                    plt.scatter(y_pred, standardized_residuals, alpha=0.6)
                    plt.xlabel('Fitted Values')
                    plt.ylabel('√|Residuals|')
                    plt.title('Scale-Location Plot')
                    
                    plt.tight_layout()
                    lav_path = 'outputs_ml/linear_assumptions_validation.png'
                    plt.savefig(lav_path, dpi=300, bbox_inches='tight')
                    mlflow.log_artifact(lav_path)
                    plt.close()
                    
                    # Log assumption validation metrics
                    mlflow.log_metric("residuals_mean", np.mean(residuals))
                    mlflow.log_metric("residuals_std", np.std(residuals))
                    
                    # Durbin-Watson test for autocorrelation
                    dw_stat = durbin_watson(residuals)
                    mlflow.log_metric("durbin_watson_stat", dw_stat)
                    
                    print(f"    Residuals mean: {np.mean(residuals):.4f}")
                    print(f"    Residuals std: {np.std(residuals):.4f}")
                    print(f"    Durbin-Watson: {dw_stat:.4f}")
                    
                except Exception as e:
                    print(f"Linear assumption validation skipped: {e}")

            print("Advanced analysis complete.\n")

if __name__ == "__main__":
    # 1. Create config and pipeline instance
    config = Config()
    pipeline = ModelDevelopmentPipeline(config)
    
    # 2. Run steps
    pipeline.load_and_prepare_data()
    pipeline.split_and_scale_data()
    pipeline.train_and_evaluate_models()
    
    # 3. Compare models and get the best
    best_model_name = pipeline.compare_models()
    
    # 4. Advanced analysis (includes SHAP)
    pipeline.advanced_model_analysis(best_model_name)
