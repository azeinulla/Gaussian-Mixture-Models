from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

from sklearn.decomposition import PCA

def perform_pca(df, variables, n_components= 2 ):
    """
    Performs PCA on the given variables in the DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data.
    - variables: list of column names for PCA.
    - n_components: number of components to keep, or the fraction of variance to explain (default: 0.95).

    Returns:
    - pca_data: DataFrame of transformed principal components.
    - pca_model: Fitted PCA model for further analysis.
    """
    # Extract and standardize the data
    clus_data = df[variables].dropna()
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(clus_data)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_transformed = pca.fit_transform(standardized_data)

    # Create a DataFrame for the PCA components
    pca_data = pd.DataFrame(pca_transformed, index=clus_data.index, 
                            columns=[f'PC{i+1}' for i in range(pca_transformed.shape[1])])

    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_)}")
    
    return pca_data, pca

pca_data, pca_model = perform_pca(df, variables=['Var1','Var2', 'Var3','Var4', 'Var5'], n_components=2 )


from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


def perform_gmm_clustering(pca_data, n_clusters=4):
    """
    Performs Gaussian Mixture Model clustering on the PCA-transformed data.

    Parameters:
    - pca_data: DataFrame of PCA-transformed components.
    - n_clusters: Number of clusters for the GMM (default: 3).

    Returns:
    - gmm_labels: Cluster labels assigned by GMM.
    - gmm: Fitted GMM model.
    """
    # Fit GMM model
    gmm = GaussianMixture(n_components=n_clusters, covariance_type= 'tied', random_state=42)
    gmm.fit(pca_data)
    
    # Predict cluster labels
    gmm_labels = gmm.predict(pca_data)
    pca_data['Cluster'] = gmm_labels
    df['Cluster'] =  gmm_labels
    
    # Print cluster information
    print(f"GMM converged: {gmm.converged_}")
    print(f"GMM means:\n{gmm.means_}")
    
    return gmm_labels, gmm

# Example Usage
n_clusters = 4  # You can adjust this based on your data
gmm_labels, gmm_model = perform_gmm_clustering(pca_data, n_clusters=n_clusters)

# Evaluate Silhouette Score
silhouette_avg = silhouette_score(pca_data[['PC1', 'PC2']], gmm_labels)
print(f"Silhouette Score for GMM with {n_clusters} clusters: {silhouette_avg:.4f}")

# Visualize Clusters in PCA Space
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_data, palette='Set1', s=30)
plt.title(f'GMM Clustering with {n_clusters} Clusters', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)
plt.legend(title='Cluster')
plt.grid()
plt.show()
