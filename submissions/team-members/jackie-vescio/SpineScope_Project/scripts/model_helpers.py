import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import learning_curve

# --- Confusion Matrix Plot Helper ---
def plot_cm(y_true, y_pred, labels=None, title='Confusion Matrix'):
    """
    Displays a confusion matrix heatmap.
    labels: optional iterable of label names to display (e.g., le.classes_)
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()


# --- Learning Curve Plot Helper ---
def plot_learning_curve(estimator, X, y, title='Learning Curve', cv=5, scoring='accuracy'):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.title(title)
    plt.xlabel('Training examples')
    plt.ylabel(scoring.capitalize())
    plt.legend(loc='best')
    plt.grid()
    plt.show()

# --- Data Loader Helper ---
def load_data(filename, subfolder='data'):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, subfolder, filename)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ File not found: {data_path}")

    print(f"✅ Loading data from: {data_path}")
    return pd.read_csv(data_path)
