# Neural Network Implementation for Orthopedic Patient Classification

## Overview
This implementation provides a comprehensive neural network solution for orthopedic patient classification using PyTorch. It includes a feedforward neural network with advanced features like batch normalization, dropout, early stopping, and comparison with tree-based methods.

## Features

### Neural Network Architecture
- **Feedforward Neural Network (FFNN)** with customizable layers
- **Dense layers** with configurable sizes
- **Dropout regularization** to prevent overfitting
- **Batch normalization** for stable training
- **Multiple activation functions** (ReLU, Tanh, Sigmoid, Leaky ReLU)
- **Xavier/He weight initialization**

### Training Features
- **Early stopping** to prevent overfitting
- **Learning rate scheduling** (ReduceLROnPlateau)
- **Class imbalance handling** with SMOTE
- **Robust data preprocessing** with StandardScaler
- **Cross-validation support**

### Evaluation & Analysis
- **Comprehensive metrics** (Accuracy, Precision, Recall, F1, ROC-AUC)
- **Residual analysis** for regression tasks
- **Confusion matrix visualization**
- **ROC and Precision-Recall curves**
- **Training history plots**

### MLflow Integration
- **Experiment tracking** for all hyperparameters
- **Model versioning** and artifact logging
- **Metric logging** throughout training
- **Model comparison** and performance tracking

### Model Comparison
- **Random Forest** baseline
- **Gradient Boosting** comparison
- **LightGBM** (optional)
- **CatBoost** (optional)
- **Performance visualization**

## Installation

1. Install required dependencies:
```bash
pip install -r nn_requirements.txt
```

2. Optional: Install additional tree-based models:
```bash
pip install lightgbm catboost
```

## Usage

### Basic Usage
```python
from nn_model import NeuralNetworkPipeline, Config

# Create configuration
config = Config()

# Initialize pipeline
pipeline = NeuralNetworkPipeline(config)

# Run complete pipeline
nn_metrics, comparison_models = pipeline.run_full_pipeline()
```

### Custom Configuration
```python
# Customize neural network architecture
config = Config()
config.HIDDEN_LAYERS = [256, 128, 64, 32]  # Custom layer sizes
config.DROPOUT_RATE = 0.4                   # Higher dropout
config.BATCH_SIZE = 64                       # Larger batch size
config.LEARNING_RATE = 0.0005               # Lower learning rate
config.MAX_EPOCHS = 300                      # More epochs
config.PATIENCE = 30                         # More patience for early stopping

# Run with custom config
pipeline = NeuralNetworkPipeline(config)
pipeline.run_full_pipeline()
```

### Running Individual Components
```python
# Load and prepare data
pipeline.load_and_prepare_data()
pipeline.split_and_scale_data()

# Create and train model
pipeline.create_data_loaders()
pipeline.create_model()
pipeline.train_model()

# Evaluate model
metrics = pipeline.evaluate_model()

# Train comparison models
comparison_results = pipeline.train_comparison_models()
```

## Configuration Options

### Neural Network Architecture
```python
HIDDEN_LAYERS = [128, 64, 32]    # Hidden layer sizes
DROPOUT_RATE = 0.3               # Dropout probability
BATCH_NORM = True                # Use batch normalization
ACTIVATION = "relu"              # Activation function
```

### Training Parameters
```python
BATCH_SIZE = 32                  # Training batch size
MAX_EPOCHS = 200                 # Maximum training epochs
LEARNING_RATE = 0.001            # Initial learning rate
WEIGHT_DECAY = 1e-5              # L2 regularization
PATIENCE = 20                    # Early stopping patience
```

### Data Processing
```python
RANDOM_STATE = 42                # Random seed
TEST_SIZE = 0.2                  # Test set proportion
VALIDATION_SIZE = 0.2            # Validation set proportion
USE_SMOTE = True                 # Apply SMOTE for class balance
CLASS_WEIGHTS = True             # Use class weights in loss
```

### MLflow Settings
```python
EXPERIMENT_NAME = "Neural_Network_Orthopedic_Classification"
TRACKING_URI = "sqlite:///mlflow_nn.db"
ARTIFACT_ROOT = "./mlruns_nn"
```

## Output Files

The pipeline generates several output files:

1. **training_history.png** - Training and validation loss/accuracy plots
2. **nn_classification_evaluation.png** - Confusion matrix, ROC curve, PR curve
3. **model_comparison.png** - Performance comparison between models
4. **MLflow artifacts** - Model files, metrics, and parameters

## MLflow Tracking

### Starting MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow_nn.db
```

### Tracked Information
- **Hyperparameters**: Architecture, learning rate, batch size, etc.
- **Metrics**: Training/validation loss, accuracy, precision, recall, F1
- **Models**: Trained neural network and comparison models
- **Artifacts**: Plots, training history, model summaries

## Model Architecture

### Default Configuration
```
Input Layer (6 features)
    ↓
Hidden Layer 1 (128 neurons)
    ↓ ReLU + BatchNorm + Dropout(0.3)
Hidden Layer 2 (64 neurons)
    ↓ ReLU + BatchNorm + Dropout(0.3)
Hidden Layer 3 (32 neurons)
    ↓ ReLU + BatchNorm + Dropout(0.3)
Output Layer (2 classes)
```

### Key Features
- **Batch Normalization**: Stabilizes training and accelerates convergence
- **Dropout**: Prevents overfitting by randomly setting neurons to zero
- **Early Stopping**: Stops training when validation loss stops improving
- **Learning Rate Scheduling**: Reduces learning rate when validation loss plateaus

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Average Precision**: Area under the PR curve

### Visualization
- **Confusion Matrix**: True vs predicted classifications
- **ROC Curve**: True positive rate vs false positive rate
- **Precision-Recall Curve**: Precision vs recall at different thresholds
- **Training History**: Loss and accuracy over epochs

## Comparison Models

The pipeline includes comparison with traditional tree-based methods:

1. **Random Forest**: Ensemble of decision trees
2. **Gradient Boosting**: Sequential tree boosting
3. **LightGBM**: Gradient boosting framework (optional)
4. **CatBoost**: Gradient boosting with categorical features (optional)

## Best Practices

### Data Preprocessing
- Features are standardized using StandardScaler
- Class imbalance is handled with SMOTE
- Train/validation/test splits are stratified

### Model Training
- Early stopping prevents overfitting
- Learning rate scheduling improves convergence
- Batch normalization stabilizes training
- Dropout provides regularization

### Evaluation
- Multiple metrics provide comprehensive assessment
- Residual analysis validates model assumptions
- Comparison with baselines ensures neural network adds value

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `config.BATCH_SIZE = 16`
   - Reduce model size: `config.HIDDEN_LAYERS = [64, 32]`

2. **Overfitting**
   - Increase dropout: `config.DROPOUT_RATE = 0.5`
   - Reduce model complexity: `config.HIDDEN_LAYERS = [32, 16]`
   - Increase weight decay: `config.WEIGHT_DECAY = 1e-4`

3. **Underfitting**
   - Increase model size: `config.HIDDEN_LAYERS = [256, 128, 64]`
   - Reduce dropout: `config.DROPOUT_RATE = 0.1`
   - Increase learning rate: `config.LEARNING_RATE = 0.01`

4. **Slow Training**
   - Increase batch size: `config.BATCH_SIZE = 64`
   - Use GPU if available (automatic detection)
   - Reduce patience: `config.PATIENCE = 10`

### Performance Tips
- Use GPU for faster training (automatically detected)
- Adjust batch size based on available memory
- Monitor training plots for signs of overfitting/underfitting
- Use MLflow to track experiments and compare configurations

## Example Output

```
Neural Network Pipeline
======================

Dataset shape: (310, 7)
Target distribution:
Normal      100
Abnormal    210

Training set: (198, 6)
Validation set: (50, 6)
Test set: (62, 6)

Model architecture:
FeedforwardNeuralNetwork(
  (network): Sequential(
    (0): Linear(in_features=6, out_features=128, bias=True)
    (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Dropout(p=0.3, inplace=False)
    ...
  )
)

Training...
Epoch  10: Train Loss: 0.4523, Train Acc: 78.23%, Val Loss: 0.3847, Val Acc: 82.00%
Epoch  20: Train Loss: 0.3245, Train Acc: 85.35%, Val Loss: 0.3021, Val Acc: 86.00%
...
Early stopping at epoch 45

Neural Network Test Results:
  Accuracy: 0.8710
  Precision: 0.8756
  Recall: 0.8710
  F1-Score: 0.8725
  ROC-AUC: 0.9234

Random Forest Results:
  Accuracy: 0.8387
  F1-Score: 0.8421

Pipeline completed successfully!
```

This implementation provides a complete solution for neural network-based classification with comprehensive evaluation, comparison, and experiment tracking capabilities.
