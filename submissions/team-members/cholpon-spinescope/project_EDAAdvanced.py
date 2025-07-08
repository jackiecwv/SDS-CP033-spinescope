import os

# Create plots folder if it does not exist
plots_folder = "plots"
os.makedirs(plots_folder, exist_ok=True)

# Helper: redirect print to both console and file
import sys
class MultiOut:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg)
    def flush(self):
        for s in self.streams:
            s.flush()
report_file = open("reports.txt", "w")
sys.stdout = MultiOut(sys.__stdout__, report_file)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, normaltest, levene
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
warnings.filterwarnings("ignore")
import kagglehub



plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# df1 = pd.read_csv("your_data.csv")

print("="*80)
print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
print("Biomechanical Features of Orthopedic Patients")
print("="*80)

# Download latest version
path = kagglehub.dataset_download("uciml/biomechanical-features-of-orthopedic-patients")
print("Path to dataset files:", path)
# Load dataset
os.listdir(path)
df = pd.read_csv(os.path.join(path, 'column_3C_weka.csv'))



def dataset_overview(df):
    print("\n" + "="*50)
    print("1. DATASET OVERVIEW")
    print("="*50)
    print(f"Dataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nColumn Information:")
    print("-" * 40)
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_vals = df[col].nunique()
        print(f"{col:25} | {str(dtype):10} | Nulls: {null_count:3} ({null_pct:5.1f}%) | Unique: {unique_vals:4}")
    print(f"\nData Types Summary:")
    print(df.dtypes.value_counts())
    if df.isnull().sum().sum() > 0:
        print("\nMissing Values Details:")
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
        })
        print(missing_data[missing_data['Missing_Count'] > 0])
    else:
        print("\n✓ No missing values found in the dataset")

def target_analysis(df, target_col='class'):
    print("\n" + "="*50)
    print("2. TARGET VARIABLE ANALYSIS")
    print("="*50)
    class_counts = df[target_col].value_counts()
    class_props = df[target_col].value_counts(normalize=True)
    print("Class Distribution:")
    print("-" * 20)
    for class_name, count in class_counts.items():
        percentage = class_props[class_name] * 100
        print(f"{class_name:15}: {count:4} ({percentage:5.1f}%)")
    max_class = class_counts.max()
    min_class = class_counts.min()
    imbalance_ratio = max_class / min_class
    print(f"\nClass Imbalance Analysis:")
    print(f"Imbalance Ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 1.5:
        print("⚠️  Dataset shows class imbalance - consider balancing techniques")
    else:
        print("✓ Dataset is relatively balanced")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.countplot(data=df, x=target_col, ax=axes[0])
    axes[0].set_title('Class Distribution (Count)')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 0.5, str(v), ha='center')
    axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Class Distribution (Proportion)')
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/target_class_distribution.png", bbox_inches="tight")
    plt.show()

def numerical_features_analysis(df, target_col='class'):
    print("\n" + "="*50)
    print("3. NUMERICAL FEATURES ANALYSIS")
    print("="*50)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("Descriptive Statistics:")
    print("-" * 25)
    desc_stats = df[numerical_cols].describe()
    print(desc_stats.round(3))
    print("\nDistribution Analysis:")
    print("-" * 25)
    distribution_stats = pd.DataFrame({
        'Feature': numerical_cols,
        'Skewness': df[numerical_cols].skew(),
        'Kurtosis': df[numerical_cols].kurtosis(),
        'Range': df[numerical_cols].max() - df[numerical_cols].min(),
        'IQR': df[numerical_cols].quantile(0.75) - df[numerical_cols].quantile(0.25)
    })
    print(distribution_stats.round(3))
    print("\nNormality Tests (Shapiro-Wilk p-values):")
    print("-" * 35)
    for col in numerical_cols:
        if len(df[col].dropna()) <= 5000:
            _, p_value = stats.shapiro(df[col].dropna())
            status = "Normal" if p_value > 0.05 else "Non-normal"
            print(f"{col:25}: p={p_value:.6f} ({status})")
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    for i, col in enumerate(numerical_cols):
        sns.histplot(data=df, x=col, hue=target_col, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].axvline(df[col].mean(), color='red', linestyle='--', alpha=0.7, label='Mean')
        axes[i].axvline(df[col].median(), color='green', linestyle='--', alpha=0.7, label='Median')
        axes[i].legend()
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/numerical_features_distribution.png", bbox_inches="tight")
    plt.show()
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    for i, col in enumerate(numerical_cols):
        sns.boxplot(data=df, x=target_col, y=col, ax=axes[i])
        axes[i].set_title(f'Box Plot: {col} by Class')
        axes[i].tick_params(axis='x', rotation=45)
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/numerical_features_boxplots.png", bbox_inches="tight")
    plt.show()

def outlier_analysis(df, target_col='class'):
    print("\n" + "="*50)
    print("4. OUTLIER DETECTION AND ANALYSIS")
    print("="*50)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_summary = []
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        z_scores = np.abs(stats.zscore(df[col]))
        z_outliers = (z_scores > 3).sum()
        median = df[col].median()
        mad = np.median(np.abs(df[col] - median))
        modified_z_scores = 0.6745 * (df[col] - median) / mad
        modified_z_outliers = (np.abs(modified_z_scores) > 3.5).sum()
        outlier_summary.append({
            'Feature': col,
            'IQR_Outliers': iqr_outliers,
            'Z_Score_Outliers': z_outliers,
            'Modified_Z_Outliers': modified_z_outliers,
            'IQR_Percentage': (iqr_outliers / len(df)) * 100,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound
        })
    outlier_df = pd.DataFrame(outlier_summary)
    print("Outlier Detection Summary:")
    print("-" * 30)
    print(outlier_df.round(3))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    methods = ['IQR_Outliers', 'Z_Score_Outliers', 'Modified_Z_Outliers']
    for i, method in enumerate(methods):
        if i < 3:
            ax = axes[i//2, i%2]
            sns.barplot(data=outlier_df, x='Feature', y=method, ax=ax)
            ax.set_title(f'Outliers by {method.replace("_", " ")}')
            ax.tick_params(axis='x', rotation=45)
    sns.barplot(data=outlier_df, x='Feature', y='IQR_Percentage', ax=axes[1, 1])
    axes[1, 1].set_title('Outlier Percentage (IQR Method)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/outlier_detection_summary.png", bbox_inches="tight")
    plt.show()

def correlation_analysis(df, target_col='class'):
    print("\n" + "="*50)
    print("5. CORRELATION AND MULTICOLLINEARITY ANALYSIS")
    print("="*50)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numerical_cols].corr()
    print("Correlation Matrix:")
    print("-" * 20)
    print(corr_matrix.round(3))
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    if high_corr_pairs:
        print("\nHighly Correlated Pairs (|r| > 0.7):")
        print("-" * 35)
        for pair in high_corr_pairs:
            print(f"{pair[0]} <-> {pair[1]}: r = {pair[2]:.3f}")
    else:
        print("\nNo highly correlated pairs found (|r| > 0.7)")
    print("\nVariance Inflation Factor (VIF) Analysis:")
    print("-" * 40)
    X_numeric = df[numerical_cols].dropna()
    vif_data = pd.DataFrame({
        "Feature": numerical_cols,
        "VIF": [variance_inflation_factor(X_numeric.values, i) for i in range(X_numeric.shape[1])]
    })
    print(vif_data.round(3))
    high_vif = vif_data[vif_data['VIF'] > 10]
    if not high_vif.empty:
        print("\n⚠️  Features with high VIF (>10) indicating multicollinearity:")
        print(high_vif)
    else:
        print("\n✓ No severe multicollinearity detected (all VIF < 10)")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, ax=axes[0, 0], fmt='.2f')
    axes[0, 0].set_title('Correlation Matrix')
    if target_col in numerical_cols:
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
        sns.barplot(x=target_corr.values, y=target_corr.index, ax=axes[0, 1])
        axes[0, 1].set_title(f'Correlation with {target_col}')
        axes[0, 1].set_xlabel('Correlation Coefficient')
    sns.barplot(data=vif_data, x='VIF', y='Feature', ax=axes[1, 0])
    axes[1, 0].set_title('Variance Inflation Factor (VIF)')
    axes[1, 0].axvline(x=10, color='red', linestyle='--', alpha=0.7, label='VIF=10')
    axes[1, 0].legend()
    plt.savefig(f"{plots_folder}/correlation_multicollinearity_matrix.png", bbox_inches="tight")
    sns.clustermap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, figsize=(8, 8))
    plt.savefig(f"{plots_folder}/correlation_clustermap.png", bbox_inches="tight")
    plt.show()
    plt.tight_layout()
    plt.show()

def feature_relationships(df, target_col='class'):
    print("\n" + "="*50)
    print("6. FEATURE RELATIONSHIPS AND INTERACTIONS")
    print("="*50)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("Generating pair plot for feature relationships...")
    if len(numerical_cols) <= 6:
        g = sns.pairplot(df, hue=target_col, diag_kind='hist', plot_kws={'alpha': 0.6})
        g.savefig(f"{plots_folder}/feature_pairplot.png", bbox_inches="tight")
        plt.show()
    else:
        print("Too many features for pair plot. Showing correlation matrix instead.")
    print("\nFeature Interactions Analysis:")
    print("-" * 35)
    le = LabelEncoder()
    df_encoded = df.copy()
    if df[target_col].dtype == 'object':
        df_encoded[target_col] = le.fit_transform(df[target_col])
    interaction_results = []
    for col in numerical_cols:
        groups = [df[df[target_col] == group][col].dropna() for group in df[target_col].unique()]
        if len(groups) > 1 and all(len(group) > 1 for group in groups):
            f_stat, p_value = stats.f_oneway(*groups)
            effect_size = f_stat / (f_stat + sum(len(group) for group in groups) - len(groups))
            interaction_results.append({
                'Feature': col,
                'F_Statistic': f_stat,
                'P_Value': p_value,
                'Effect_Size': effect_size,
                'Significant': p_value < 0.05
            })
    if interaction_results:
        interaction_df = pd.DataFrame(interaction_results)
        print(interaction_df.round(6))
        significant_features = interaction_df[interaction_df['Significant']]['Feature'].tolist()
        if significant_features:
            n_features = len(significant_features)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            for i, feature in enumerate(significant_features):
                sns.violinplot(data=df, x=target_col, y=feature, ax=axes[i])
                axes[i].set_title(f'{feature} by {target_col}')
                axes[i].tick_params(axis='x', rotation=45)
            for i in range(len(significant_features), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            plt.savefig(f"{plots_folder}/feature_interactions_violin.png", bbox_inches="tight")
            plt.show()

def dimensionality_analysis(df, target_col='class'):
    print("\n" + "="*50)
    print("7. DIMENSIONALITY REDUCTION ANALYSIS")
    print("="*50)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numerical_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)
    print("PCA Analysis:")
    print("-" * 15)
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_.round(4)}")
    print(f"Cumulative Explained Variance: {pca.explained_variance_ratio_.cumsum().round(4)}")
    cumsum_var = pca.explained_variance_ratio_.cumsum()
    n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
    print(f"Components needed for 95% variance: {n_components_95}")
    print("\nPerforming t-SNE analysis...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(X_scaled)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes[0, 0].plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, 'bo-', label='Individual')
    axes[0, 0].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'ro-', label='Cumulative')
    axes[0, 0].axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label='95% Variance')
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].set_title('PCA Explained Variance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    colors = plt.cm.Set1(np.linspace(0, 1, len(df[target_col].unique())))
    for i, target_class in enumerate(df[target_col].unique()):
        mask = df[target_col] == target_class
        axes[0, 1].scatter(pca_result[mask, 0], pca_result[mask, 1], c=[colors[i]], label=target_class, alpha=0.6)
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[0, 1].set_title('PCA 2D Projection')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    for i, target_class in enumerate(df[target_col].unique()):
        mask = df[target_col] == target_class
        axes[1, 0].scatter(tsne_result[mask, 0], tsne_result[mask, 1], c=[colors[i]], label=target_class, alpha=0.6)
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    axes[1, 0].set_title('t-SNE 2D Projection')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    if len(numerical_cols) <= 10:
        loadings = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])
        loading_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=numerical_cols)
        for i, feature in enumerate(numerical_cols):
            axes[1, 1].arrow(0, 0, loading_df.loc[feature, 'PC1'], loading_df.loc[feature, 'PC2'], head_width=0.05, head_length=0.05, fc='red', ec='red')
            axes[1, 1].text(loading_df.loc[feature, 'PC1'] * 1.1, loading_df.loc[feature, 'PC2'] * 1.1, feature, fontsize=8, ha='center', va='center')
        axes[1, 1].set_xlabel('PC1 Loading')
        axes[1, 1].set_ylabel('PC2 Loading')
        axes[1, 1].set_title('PCA Loading Plot')
        axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/dimensionality_reduction.png", bbox_inches="tight")
    plt.show()

def clustering_analysis(df, target_col='class'):
    print("\n" + "="*50)
    print("8. CLUSTERING ANALYSIS")
    print("="*50)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numerical_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    inertias = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    from sklearn.metrics import silhouette_score
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    optimal_k = range(2, 11)[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters (silhouette): {optimal_k}")
    print(f"Best silhouette score: {max(silhouette_scores):.3f}")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_cluster = df.copy()
    df_cluster['Cluster'] = cluster_labels
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes[0, 0].plot(k_range, inertias, 'bo-')
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].set_title('Elbow Method for Optimal k')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(range(2, 11), silhouette_scores, 'ro-')
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette Analysis')
    axes[0, 1].grid(True, alpha=0.3)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    scatter = axes[1, 0].scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    axes[1, 0].set_xlabel('First Principal Component')
    axes[1, 0].set_ylabel('Second Principal Component')
    axes[1, 0].set_title('Clusters in PCA Space')
    plt.colorbar(scatter, ax=axes[1, 0])
    cluster_crosstab = pd.crosstab(df_cluster['Cluster'], df_cluster[target_col])
    sns.heatmap(cluster_crosstab, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_title('Clusters vs True Labels')
    axes[1, 1].set_xlabel('True Labels')
    axes[1, 1].set_ylabel('Clusters')
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/clustering_analysis.png", bbox_inches="tight")
    plt.show()
    print("\nCluster vs True Labels Cross-tabulation:")
    print(cluster_crosstab)

def statistical_tests(df, target_col='class'):
    print("\n" + "="*50)
    print("9. STATISTICAL TESTS")
    print("="*50)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    groups = df.groupby(target_col)
    group_names = list(groups.groups.keys())
    group_data = [df[df[target_col] == name][numerical_cols] for name in group_names]
    stats_results = []
    for col in numerical_cols:
        print(f"\nFeature: {col}")
        print("-" * 20)
        feat_group_data = [df[df[target_col] == name][col].dropna() for name in group_names]
        normality = []
        for i, data in enumerate(feat_group_data):
            if len(data) > 3:
                _, pval = stats.shapiro(data)
                normality.append(pval > 0.05)
                print(f"  {group_names[i]}: Shapiro p={pval:.4f} ({'Normal' if pval>0.05 else 'Non-normal'})")
            else:
                normality.append(False)
                print(f"  {group_names[i]}: Not enough data for normality test")
        if all([len(x)>3 for x in feat_group_data]):
            _, p_levene = levene(*feat_group_data)
            print(f"  Levene’s p={p_levene:.4f} ({'Equal variances' if p_levene>0.05 else 'Unequal variances'})")
        else:
            p_levene = np.nan
        if all(normality) and (p_levene > 0.05):
            f_stat, p_anova = stats.f_oneway(*feat_group_data)
            print(f"  ANOVA F={f_stat:.3f}, p={p_anova:.4e}")
            test_type = "ANOVA"
            test_stat = f_stat
            p_val = p_anova
        else:
            h_stat, p_kruskal = stats.kruskal(*feat_group_data)
            print(f"  Kruskal-Wallis H={h_stat:.3f}, p={p_kruskal:.4e}")
            test_type = "Kruskal-Wallis"
            test_stat = h_stat
            p_val = p_kruskal
        stats_results.append({
            'Feature': col,
            'Test': test_type,
            'Test Statistic': test_stat,
            'p-value': p_val,
            'Significant': p_val < 0.05
        })
    results_df = pd.DataFrame(stats_results)
    print("\nSummary Table:")
    print(results_df.round(5))
    significant = results_df[results_df['Significant']]
    if not significant.empty:
        print("\nFeatures with significant class differences (p < 0.05):")
        print(significant[['Feature', 'Test', 'p-value']])
    else:
        print("\nNo statistically significant feature differences found at p < 0.05.")
    if not significant.empty:
        print("\nBoxplots for features with significant class differences:")
        for col in significant['Feature']:
            plt.figure(figsize=(6, 4))
            sns.boxplot(data=df, x=target_col, y=col)
            plt.title(f'{col} by {target_col}')
            plt.tight_layout()
            plt.savefig(f"{plots_folder}/statistical_test_{col}_boxplot.png", bbox_inches="tight")
            plt.show()

# End of script: close the report file properly
# (You may want to do this at the end of your notebook or after calling all functions)
# report_file.close()


# RUN THE ANALYSIS
dataset_overview(df)
target_analysis(df, target_col='class')
numerical_features_analysis(df, target_col='class')
outlier_analysis(df, target_col='class')
correlation_analysis(df, target_col='class')
feature_relationships(df, target_col='class')
dimensionality_analysis(df, target_col='class')
clustering_analysis(df, target_col='class')
statistical_tests(df, target_col='class')

report_file.close()



