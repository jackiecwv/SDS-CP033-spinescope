import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
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
import kagglehub
from datetime import datetime
import json
import pickle
from typing import Dict, List, Tuple, Any
import shap
from sklearn.model_selection import learning_curve
from imblearn.over_sampling import SMOTE   #

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
        self.label_encoder = None
        self.feature_names = None
        self.models_performance = {}
        
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
        
        # Download dataset
        path = kagglehub.dataset_download("uciml/biomechanical-features-of-orthopedic-patients")
        self.df = pd.read_csv(os.path.join(path, 'column_3C_weka.csv'))
        
        # Create binary classification target
        self.df['binary_class'] = self.df['class'].replace({
            'Hernia': 'Abnormal',
            'Spondylolisthesis': 'Abnormal',
            'Normal': 'Normal'
        })
        
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
            
            with open("dataset_info.json", "w") as f:
                json.dump(dataset_info, f, indent=2)
            mlflow.log_artifact("dataset_info.json")
            os.remove("dataset_info.json")
    
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
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Validation set: {self.X_val.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        # Log data split info
        with mlflow.start_run(run_name="data_splitting"):
            mlflow.log_param("train_size", len(self.X_train))
            mlflow.log_param("val_size", len(self.X_val))
            mlflow.log_param("test_size", len(self.X_test))
            mlflow.log_param("n_features", len(self.feature_names))
            mlflow.log_param("feature_names", self.feature_names)
            mlflow.log_param("target_encoding", dict(zip(self.label_encoder.classes_, 
                                                       self.label_encoder.transform(self.label_encoder.classes_))))
    
    def get_model_configurations(self) -> Dict[str, Dict]:
        """Define model configurations with hyperparameter grids"""
        return {
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.config.RANDOM_STATE, max_iter=1000),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga'],
                    'class_weight': [None, 'balanced']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.config.RANDOM_STATE),
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None],
                    'class_weight': [None, 'balanced']
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.config.RANDOM_STATE),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'svm': {
                'model': SVC(random_state=self.config.RANDOM_STATE, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'poly', 'linear'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'class_weight': [None, 'balanced']
                }
            },
            'naive_bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': np.logspace(-10, -6, 10)
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=self.config.RANDOM_STATE),
                'params': {
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'max_features': ['sqrt', 'log2', None],
                    'class_weight': [None, 'balanced']
                }
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            'mlp': {
                'model': MLPClassifier(random_state=self.config.RANDOM_STATE, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam', 'lbfgs'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
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
    
    def create_evaluation_plots(self, model, X_test, y_test, model_name: str):
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
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'])
            axes[1, 1].set_title(f'Feature Importance - {model_name}')
            axes[1, 1].set_xlabel('Importance')
        elif hasattr(model, 'coef_'):
            coef_importance = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': np.abs(model.coef_[0])
            }).sort_values('coefficient', ascending=True)
            
            axes[1, 1].barh(coef_importance['feature'], coef_importance['coefficient'])
            axes[1, 1].set_title(f'Feature Coefficients - {model_name}')
            axes[1, 1].set_xlabel('Absolute Coefficient')
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return f'{model_name}_evaluation.png'
    
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
                    
                    # Determine if model needs scaled features
                    use_scaled = model_name in ['logistic_regression', 'svm', 'knn', 'mlp']
                    X_train_use = self.X_train_scaled if use_scaled else self.X_train
                    X_val_use = self.X_val_scaled if use_scaled else self.X_val
                    X_test_use = self.X_test_scaled if use_scaled else self.X_test
                    
                    mlflow.log_param("features_scaled", use_scaled)
                    
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
                    search.fit(X_train_use, self.y_train)
                    
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
                        'use_scaled': use_scaled
                    }
                    
                    # Create evaluation plots
                    plot_path = self.create_evaluation_plots(best_model, X_test_use, self.y_test, model_name)
                    mlflow.log_artifact(plot_path)
                    os.remove(plot_path)
                    
                    # Log model
                    signature = infer_signature(X_train_use, self.y_train)
                    mlflow.sklearn.log_model(
                        sk_model=best_model,
                        artifact_path=f"model_{model_name}",
                        signature=signature,
                        input_example=X_train_use[:5]
                    )
                    
                    # Feature importance analysis
                    if hasattr(best_model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'feature': self.feature_names,
                            'importance': best_model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        # Log top features
                        top_features = feature_importance.head(5)['feature'].tolist()
                        mlflow.log_param("top_5_features", top_features)
                        
                        # Save feature importance
                        feature_importance.to_csv(f'{model_name}_feature_importance.csv', index=False)
                        mlflow.log_artifact(f'{model_name}_feature_importance.csv')
                        os.remove(f'{model_name}_feature_importance.csv')
                    
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
            
            # Save comparison
            comparison_df.to_csv('model_comparison.csv', index=False)
            mlflow.log_artifact('model_comparison.csv')
            
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
            plt.savefig('model_comparison_chart.png', dpi=300, bbox_inches='tight')
            mlflow.log_artifact('model_comparison_chart.png')
            plt.close()
            
            # Clean up
            os.remove('model_comparison.csv')
            os.remove('model_comparison_chart.png')
            
            return best_model_name
    
    def advanced_model_analysis(self, best_model_name: str):
        """Perform advanced analysis on the best model"""
        print(f"\n" + "="*60)
        print(f"ADVANCED ANALYSIS - {best_model_name.upper()}")
        print("="*60)
        
        best_model_info = self.models_performance[best_model_name]
        best_model = best_model_info['model']
        use_scaled = best_model_info['use_scaled']
        
        X_test_use = self.X_test_scaled if use_scaled else self.X_test
        
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
            plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
            mlflow.log_artifact('learning_curves.png')
            plt.close()
            
            # 2. Permutation Importance
            print("Calculating permutation importance...")
            perm_importance = permutation_importance(
                best_model, X_test_use, self.y_test, 
                n_repeats=10, random_state=self.config.RANDOM_STATE
            )
            
            perm_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            plt.figure(figsize=(10, 6))
            plt.barh(perm_df['feature'], perm_df['importance_mean'], 
                    xerr=perm_df['importance_std'])
            plt.xlabel('Permutation Importance')
            plt.title(f'Permutation Importance - {best_model_name}')
            plt.tight_layout()
            plt.savefig('permutation_importance.png', dpi=300, bbox_inches='tight')
            mlflow.log_artifact('permutation_importance.png')
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
                plt.savefig('calibration_plot.png', dpi=300, bbox_inches='tight')
                mlflow.log_artifact('calibration_plot.png')
                plt.close()
            # 4. SHAP values (if supported)
            try:
                print("Computing SHAP values...")
                explainer = shap.Explainer(best_model, X_test_use)
                shap_values = explainer(X_test_use)
                shap.summary_plot(shap_values, X_test_use, feature_names=self.feature_names, show=False)
                plt.tight_layout()
                plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
                mlflow.log_artifact('shap_summary.png')
                plt.close()
            except Exception as e:
                print(f"SHAP analysis skipped: {e}")

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
