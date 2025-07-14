import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings("ignore")

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

# Scikit-learn imports
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, mean_squared_error,
    mean_absolute_error, r2_score
)

# Tree-based models for comparison
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# MLflow imports
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Other imports
import kagglehub
from datetime import datetime
import json
import pickle
from scipy import stats
from imblearn.over_sampling import SMOTE


class Config:
    """Configuration class for neural network pipeline"""
    # MLflow settings
    EXPERIMENT_NAME = "Neural_Network_Orthopedic_Classification"
    TRACKING_URI = "sqlite:///mlflow_nn.db"
    ARTIFACT_ROOT = "./mlruns_nn"
    
    # Data settings
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
    
    # Neural network architecture
    HIDDEN_LAYERS = [128, 64, 32]  # Hidden layer sizes
    DROPOUT_RATE = 0.3
    BATCH_NORM = True
    ACTIVATION = "relu"
    
    # Training settings
    BATCH_SIZE = 32
    MAX_EPOCHS = 200
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    PATIENCE = 20  # Early stopping patience
    
    # Device settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Task type
    TASK_TYPE = "classification"  # or "regression"
    
    # Class imbalance handling
    USE_SMOTE = True
    CLASS_WEIGHTS = True


class FeedforwardNeuralNetwork(nn.Module):
    """Feedforward Neural Network with customizable architecture"""
    
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int,
                 dropout_rate: float = 0.3, batch_norm: bool = True, 
                 activation: str = "relu", task_type: str = "classification"):
        super(FeedforwardNeuralNetwork, self).__init__()
        
        self.task_type = task_type
        self.activation = activation
        self.batch_norm = batch_norm
        
        # Define activation function
        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "tanh":
            self.activation_fn = nn.Tanh()
        elif activation == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif activation == "leaky_relu":
            self.activation_fn = nn.LeakyReLU(0.01)
        else:
            self.activation_fn = nn.ReLU()
        
        # Build the network layers
        layers = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            # Linear layer
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Don't add activation, batch norm, or dropout to the output layer
            if i < len(layer_sizes) - 2:
                # Batch normalization
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                
                # Activation function
                layers.append(self.activation_fn)
                
                # Dropout
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation == "relu":
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)
    
    def get_architecture_summary(self):
        """Get a summary of the network architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": str(self.network)
        }


class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
    
    def save_checkpoint(self, model):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()


class NeuralNetworkPipeline:
    """Comprehensive neural network pipeline with MLflow tracking"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        print(f"Using device: {self.device}")
        
        # Data containers
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
        
        # Model containers
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Comparison models
        self.comparison_models = {}
        
        # Setup MLflow
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Initialize MLflow tracking"""
        mlflow.set_tracking_uri(self.config.TRACKING_URI)
        
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
        """Load and prepare the orthopedic dataset"""
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
        
        # Log dataset info
        with mlflow.start_run(run_name="data_preparation"):
            mlflow.log_param("dataset_shape", self.df.shape)
            mlflow.log_param("n_features", len(self.df.select_dtypes(include=[np.number]).columns))
            mlflow.log_param("target_classes", list(self.df['binary_class'].unique()))
            mlflow.log_param("class_distribution", dict(self.df['binary_class'].value_counts()))
            
            # Log data quality metrics
            mlflow.log_metric("missing_values", self.df.isnull().sum().sum())
            mlflow.log_metric("duplicate_rows", self.df.duplicated().sum())
            
            mlflow.end_run()  # End the data preparation run
    
    def split_and_scale_data(self, target_col='binary_class'):
        """Split data and prepare for training"""
        print("Splitting and scaling data...")
        
        # Prepare features and target
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        X = self.df[numerical_cols]
        y = self.df[target_col]
        
        self.feature_names = numerical_cols
        
        # For classification, encode target
        if self.config.TASK_TYPE == "classification":
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y.values
        
        # Split data
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y_encoded, test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE, 
            stratify=y_encoded if self.config.TASK_TYPE == "classification" else None
        )
        
        val_size_adj = self.config.VALIDATION_SIZE / (1 - self.config.TEST_SIZE)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj,
            random_state=self.config.RANDOM_STATE,
            stratify=y_temp if self.config.TASK_TYPE == "classification" else None
        )
        
        # Handle class imbalance for classification
        if self.config.TASK_TYPE == "classification" and self.config.USE_SMOTE:
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=self.config.RANDOM_STATE)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            print(f"After SMOTE: {np.bincount(self.y_train)}")
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train_scaled.shape}")
        print(f"Validation set: {self.X_val_scaled.shape}")
        print(f"Test set: {self.X_test_scaled.shape}")
        
        # Log data split info
        with mlflow.start_run(run_name="data_splitting"):
            mlflow.log_param("train_size", len(self.X_train))
            mlflow.log_param("val_size", len(self.X_val))
            mlflow.log_param("test_size", len(self.X_test))
            mlflow.log_param("n_features", len(self.feature_names))
            mlflow.log_param("feature_names", self.feature_names)
            mlflow.log_param("use_smote", self.config.USE_SMOTE)
            mlflow.log_param("task_type", self.config.TASK_TYPE)
            
            mlflow.end_run()  # End the data splitting run
    
    def create_data_loaders(self):
        """Create PyTorch data loaders"""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(self.X_train_scaled)
        y_train_tensor = torch.LongTensor(self.y_train) if self.config.TASK_TYPE == "classification" else torch.FloatTensor(self.y_train)
        
        X_val_tensor = torch.FloatTensor(self.X_val_scaled)
        y_val_tensor = torch.LongTensor(self.y_val) if self.config.TASK_TYPE == "classification" else torch.FloatTensor(self.y_val)
        
        X_test_tensor = torch.FloatTensor(self.X_test_scaled)
        y_test_tensor = torch.LongTensor(self.y_test) if self.config.TASK_TYPE == "classification" else torch.FloatTensor(self.y_test)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def create_model(self):
        """Create the neural network model"""
        input_size = len(self.feature_names)
        
        if self.config.TASK_TYPE == "classification":
            output_size = len(np.unique(self.y_train))
        else:
            output_size = 1
        
        self.model = FeedforwardNeuralNetwork(
            input_size=input_size,
            hidden_layers=self.config.HIDDEN_LAYERS,
            output_size=output_size,
            dropout_rate=self.config.DROPOUT_RATE,
            batch_norm=self.config.BATCH_NORM,
            activation=self.config.ACTIVATION,
            task_type=self.config.TASK_TYPE
        ).to(self.device)
        
        print(f"Model architecture:")
        print(self.model)
        
        # Print model summary
        arch_summary = self.model.get_architecture_summary()
        print(f"Total parameters: {arch_summary['total_parameters']:,}")
        print(f"Trainable parameters: {arch_summary['trainable_parameters']:,}")
        
        return self.model
    
    def train_model(self):
        """Train the neural network with early stopping"""
        print("Training neural network...")
        
        with mlflow.start_run(run_name="neural_network_training"):
            # Log hyperparameters
            mlflow.log_param("hidden_layers", self.config.HIDDEN_LAYERS)
            mlflow.log_param("dropout_rate", self.config.DROPOUT_RATE)
            mlflow.log_param("batch_norm", self.config.BATCH_NORM)
            mlflow.log_param("activation", self.config.ACTIVATION)
            mlflow.log_param("batch_size", self.config.BATCH_SIZE)
            mlflow.log_param("learning_rate", self.config.LEARNING_RATE)
            mlflow.log_param("weight_decay", self.config.WEIGHT_DECAY)
            mlflow.log_param("max_epochs", self.config.MAX_EPOCHS)
            mlflow.log_param("patience", self.config.PATIENCE)
            mlflow.log_param("device", str(self.device))
            
            # Log model architecture
            arch_summary = self.model.get_architecture_summary()
            mlflow.log_param("total_parameters", arch_summary['total_parameters'])
            mlflow.log_param("trainable_parameters", arch_summary['trainable_parameters'])
            
            # Setup loss function and optimizer
            if self.config.TASK_TYPE == "classification":
                if self.config.CLASS_WEIGHTS:
                    class_weights = torch.FloatTensor([1.0, 1.0]).to(self.device)  # Adjust as needed
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.MSELoss()
            
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
            
            # Learning rate scheduler
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            
            # Early stopping
            early_stopping = EarlyStopping(patience=self.config.PATIENCE, min_delta=0.001)
            
            # Training loop
            self.train_losses = []
            self.val_losses = []
            
            for epoch in range(self.config.MAX_EPOCHS):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = self.model(data)
                    
                    if self.config.TASK_TYPE == "classification":
                        loss = criterion(output, target)
                        pred = output.argmax(dim=1)
                        train_correct += pred.eq(target).sum().item()
                    else:
                        loss = criterion(output.squeeze(), target)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_total += target.size(0)
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in self.val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.model(data)
                        
                        if self.config.TASK_TYPE == "classification":
                            loss = criterion(output, target)
                            pred = output.argmax(dim=1)
                            val_correct += pred.eq(target).sum().item()
                        else:
                            loss = criterion(output.squeeze(), target)
                        
                        val_loss += loss.item()
                        val_total += target.size(0)
                
                # Calculate averages
                train_loss /= len(self.train_loader)
                val_loss /= len(self.val_loader)
                
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                
                # Calculate accuracy for classification
                if self.config.TASK_TYPE == "classification":
                    train_acc = 100. * train_correct / train_total
                    val_acc = 100. * val_correct / val_total
                    
                    if epoch % 10 == 0:
                        print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                    
                    # Log metrics to MLflow
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                    mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                
                else:
                    if epoch % 10 == 0:
                        print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                    
                    # Log metrics to MLflow
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    mlflow.log_param("early_stopped_epoch", epoch)
                    break
            
            # Save training history plots
            self.plot_training_history()
            
            # Log final model
            signature = infer_signature(self.X_train_scaled, self.y_train)
            mlflow.pytorch.log_model(
                self.model,
                "neural_network_model",
                signature=signature
            )
            
            mlflow.end_run()  # End the training run
            
            print("Neural network training completed!")
    
    def plot_training_history(self):
        """Plot and save training history"""
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # If we have accuracy data, plot it
        if self.config.TASK_TYPE == "classification" and hasattr(self, 'train_accuracies'):
            plt.subplot(1, 2, 2)
            plt.plot(self.train_accuracies, label='Training Accuracy')
            plt.plot(self.val_accuracies, label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('training_history.png')
        plt.close()
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("Evaluating neural network model...")
        
        self.model.eval()
        test_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                if self.config.TASK_TYPE == "classification":
                    pred = output.argmax(dim=1)
                    probs = F.softmax(output, dim=1)
                    all_probabilities.extend(probs.cpu().numpy())
                    all_predictions.extend(pred.cpu().numpy())
                else:
                    all_predictions.extend(output.squeeze().cpu().numpy())
                
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        if self.config.TASK_TYPE == "classification":
            accuracy = accuracy_score(all_targets, all_predictions)
            precision = precision_score(all_targets, all_predictions, average='weighted')
            recall = recall_score(all_targets, all_predictions, average='weighted')
            f1 = f1_score(all_targets, all_predictions, average='weighted')
            
            # ROC-AUC for binary classification
            if len(np.unique(all_targets)) == 2:
                probs_positive = np.array(all_probabilities)[:, 1]
                roc_auc = roc_auc_score(all_targets, probs_positive)
                avg_precision = average_precision_score(all_targets, probs_positive)
            else:
                roc_auc = roc_auc_score(all_targets, all_probabilities, multi_class='ovr')
                avg_precision = None
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'avg_precision': avg_precision
            }
            
            print(f"Neural Network Test Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")
            if avg_precision:
                print(f"  Avg Precision: {avg_precision:.4f}")
            
            # Log metrics to MLflow
            with mlflow.start_run(run_name="neural_network_evaluation"):
                for metric_name, metric_value in metrics.items():
                    if metric_value is not None:
                        mlflow.log_metric(f"test_{metric_name}", metric_value)
                
                mlflow.end_run()  # End the neural network evaluation run
            
            # Create evaluation plots
            self.create_classification_plots(all_targets, all_predictions, all_probabilities)
        
        else:
            # Regression metrics
            mse = mean_squared_error(all_targets, all_predictions)
            mae = mean_absolute_error(all_targets, all_predictions)
            r2 = r2_score(all_targets, all_predictions)
            rmse = np.sqrt(mse)
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2
            }
            
            print(f"Neural Network Test Results:")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
            
            # Log metrics to MLflow
            with mlflow.start_run(run_name="neural_network_evaluation"):
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", metric_value)
                
                mlflow.end_run()  # End the neural network evaluation run
            
            # Create regression plots
            self.create_regression_plots(all_targets, all_predictions)
        
        return metrics
    
    def create_classification_plots(self, y_true, y_pred, y_proba):
        """Create classification evaluation plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix - Neural Network')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # ROC Curve (for binary classification)
        if len(np.unique(y_true)) == 2:
            probs_positive = np.array(y_proba)[:, 1]
            fpr, tpr, _ = roc_curve(y_true, probs_positive)
            auc_score = roc_auc_score(y_true, probs_positive)
            axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            axes[0, 1].plot([0, 1], [0, 1], 'k--')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve - Neural Network')
            axes[0, 1].legend()
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, probs_positive)
            avg_precision = average_precision_score(y_true, probs_positive)
            axes[1, 0].plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title('Precision-Recall Curve - Neural Network')
            axes[1, 0].legend()
        
        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        axes[1, 1].bar(unique, counts)
        axes[1, 1].set_title('True Class Distribution')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('nn_classification_evaluation.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('nn_classification_evaluation.png')
        plt.close()
    
    def create_regression_plots(self, y_true, y_pred):
        """Create regression evaluation plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Predicted vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Predicted vs Actual - Neural Network')
        
        # Residuals plot
        residuals = np.array(y_true) - np.array(y_pred)
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot - Neural Network')
        
        # Residuals histogram
        axes[1, 0].hist(residuals, bins=30, density=True, alpha=0.7)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Distribution of Residuals')
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        
        plt.tight_layout()
        plt.savefig('nn_regression_evaluation.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('nn_regression_evaluation.png')
        plt.close()
    
    def train_comparison_models(self):
        """Train tree-based models for comparison"""
        print("Training comparison models...")
        
        # Ensure any active runs are ended
        try:
            mlflow.end_run()
        except:
            pass
        
        comparison_results = {}
        
        # Random Forest
        if self.config.TASK_TYPE == "classification":
            rf_model = RandomForestClassifier(n_estimators=100, random_state=self.config.RANDOM_STATE)
        else:
            rf_model = RandomForestRegressor(n_estimators=100, random_state=self.config.RANDOM_STATE)
        
        rf_model.fit(self.X_train_scaled, self.y_train)
        comparison_results['Random Forest'] = self.evaluate_comparison_model(rf_model, "Random Forest")
        
        # Gradient Boosting
        if self.config.TASK_TYPE == "classification":
            gb_model = GradientBoostingClassifier(n_estimators=100, random_state=self.config.RANDOM_STATE)
        else:
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=self.config.RANDOM_STATE)
        
        gb_model.fit(self.X_train_scaled, self.y_train)
        comparison_results['Gradient Boosting'] = self.evaluate_comparison_model(gb_model, "Gradient Boosting")
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            if self.config.TASK_TYPE == "classification":
                lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=self.config.RANDOM_STATE, verbose=-1)
            else:
                lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=self.config.RANDOM_STATE, verbose=-1)
            
            lgb_model.fit(self.X_train_scaled, self.y_train)
            comparison_results['LightGBM'] = self.evaluate_comparison_model(lgb_model, "LightGBM")
        
        # CatBoost
        if CATBOOST_AVAILABLE:
            if self.config.TASK_TYPE == "classification":
                cb_model = cb.CatBoostClassifier(n_estimators=100, random_state=self.config.RANDOM_STATE, verbose=False)
            else:
                cb_model = cb.CatBoostRegressor(n_estimators=100, random_state=self.config.RANDOM_STATE, verbose=False)
            
            cb_model.fit(self.X_train_scaled, self.y_train)
            comparison_results['CatBoost'] = self.evaluate_comparison_model(cb_model, "CatBoost")
        
        self.comparison_models = comparison_results
        return comparison_results
    
    def evaluate_comparison_model(self, model, model_name):
        """Evaluate a comparison model"""
        with mlflow.start_run(run_name=f"{model_name}_comparison"):
            y_pred = model.predict(self.X_test_scaled)
            
            if self.config.TASK_TYPE == "classification":
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted')
                recall = recall_score(self.y_test, y_pred, average='weighted')
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(self.X_test_scaled)
                    if len(np.unique(self.y_test)) == 2:
                        roc_auc = roc_auc_score(self.y_test, y_proba[:, 1])
                    else:
                        roc_auc = roc_auc_score(self.y_test, y_proba, multi_class='ovr')
                else:
                    roc_auc = None
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc
                }
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    if metric_value is not None:
                        mlflow.log_metric(f"test_{metric_name}", metric_value)
                
                print(f"{model_name} Results:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                if roc_auc:
                    print(f"  ROC-AUC: {roc_auc:.4f}")
            
            else:
                mse = mean_squared_error(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                
                metrics = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2_score': r2
                }
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", metric_value)
                
                print(f"{model_name} Results:")
                print(f"  MSE: {mse:.4f}")
                print(f"  MAE: {mae:.4f}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  R²: {r2:.4f}")
            
            # Log model
            mlflow.sklearn.log_model(model, f"{model_name.lower().replace(' ', '_')}_model")
            
            return metrics
    
    def create_model_comparison_plot(self):
        """Create comparison plot between neural network and tree-based models"""
        if not self.comparison_models:
            return
        
        # Get neural network results
        nn_metrics = self.evaluate_model()
        
        # Prepare data for comparison
        models = ['Neural Network'] + list(self.comparison_models.keys())
        
        if self.config.TASK_TYPE == "classification":
            accuracy_scores = [nn_metrics['accuracy']] + [self.comparison_models[model]['accuracy'] for model in self.comparison_models.keys()]
            f1_scores = [nn_metrics['f1_score']] + [self.comparison_models[model]['f1_score'] for model in self.comparison_models.keys()]
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy comparison
            bars1 = axes[0].bar(models, accuracy_scores)
            axes[0].set_title('Model Accuracy Comparison')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_ylim([0, 1])
            
            # Add value labels on bars
            for bar, score in zip(bars1, accuracy_scores):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
            
            # F1-Score comparison
            bars2 = axes[1].bar(models, f1_scores)
            axes[1].set_title('Model F1-Score Comparison')
            axes[1].set_ylabel('F1-Score')
            axes[1].set_ylim([0, 1])
            
            # Add value labels on bars
            for bar, score in zip(bars2, f1_scores):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        else:
            r2_scores = [nn_metrics['r2_score']] + [self.comparison_models[model]['r2_score'] for model in self.comparison_models.keys()]
            rmse_scores = [nn_metrics['rmse']] + [self.comparison_models[model]['rmse'] for model in self.comparison_models.keys()]
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # R² comparison
            bars1 = axes[0].bar(models, r2_scores)
            axes[0].set_title('Model R² Comparison')
            axes[0].set_ylabel('R² Score')
            
            # Add value labels on bars
            for bar, score in zip(bars1, r2_scores):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
            
            # RMSE comparison
            bars2 = axes[1].bar(models, rmse_scores)
            axes[1].set_title('Model RMSE Comparison')
            axes[1].set_ylabel('RMSE')
            
            # Add value labels on bars
            for bar, score in zip(bars2, rmse_scores):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('model_comparison.png')
        plt.close()
    
    def run_full_pipeline(self):
        """Run the complete neural network pipeline"""
        print("Starting Neural Network Pipeline...")
        print("="*50)
        
        # 1. Load and prepare data
        self.load_and_prepare_data()
        
        # 2. Split and scale data
        self.split_and_scale_data()
        
        # 3. Create data loaders
        self.create_data_loaders()
        
        # 4. Create model
        self.create_model()
        
        # 5. Train model
        self.train_model()
        
        # 6. Evaluate model
        nn_metrics = self.evaluate_model()
        
        # 7. Train comparison models
        self.train_comparison_models()
        
        # 8. Create comparison plots
        self.create_model_comparison_plot()
        
        print("="*50)
        print("Pipeline completed successfully!")
        
        return nn_metrics, self.comparison_models


def main():
    """Main function to run the neural network pipeline"""
    # Check for required packages
    missing_packages = []
    
    if not LIGHTGBM_AVAILABLE:
        missing_packages.append("lightgbm")
    
    if not CATBOOST_AVAILABLE:
        missing_packages.append("catboost")
    
    if missing_packages:
        print(f"Optional packages not installed: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        print("Pipeline will continue without these models.\n")
    
    # Create configuration
    config = Config()
    
    # Initialize and run pipeline
    pipeline = NeuralNetworkPipeline(config)
    
    try:
        nn_metrics, comparison_models = pipeline.run_full_pipeline()
        
        # Print final summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        
        if config.TASK_TYPE == "classification":
            print(f"Neural Network - Accuracy: {nn_metrics['accuracy']:.4f}, F1: {nn_metrics['f1_score']:.4f}")
            for model_name, metrics in comparison_models.items():
                print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        else:
            print(f"Neural Network - R²: {nn_metrics['r2_score']:.4f}, RMSE: {nn_metrics['rmse']:.4f}")
            for model_name, metrics in comparison_models.items():
                print(f"{model_name} - R²: {metrics['r2_score']:.4f}, RMSE: {metrics['rmse']:.4f}")
        
        print("\nCheck MLflow UI for detailed experiment tracking:")
        print(f"mlflow ui --backend-store-uri {config.TRACKING_URI}")
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
