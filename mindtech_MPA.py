
# Workplace Mental Health Risk Analysis

# Reason for Model Choice: KMeans is ideal for grouping similar workplaces based on their mental health support policies. It provides interpretable clusters and helps in identifying high-risk or low-support workplace groups.


import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging

# Setup logging
# Logging is crucial to track the progress and pinpoint issues during execution. Here, logs are written to a file named
# 'clustering_log.txt' to keep a detailed record of operations, errors, and intermediate results for reproducibility.
log_file = "clustering_log.txt"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting clustering script execution.")

# Load the dataset
# The dataset is loaded dynamically using os to ensure compatibility across operating systems and environments.
# This approach is particularly useful when deploying the script on cloud platforms or sharing across team members.
data_path = os.path.join(os.getcwd(), 'survey.csv')
data = pd.read_csv(data_path)
logging.info("Dataset loaded successfully.")

# Data Cleaning
# Missing values in columns 'self_employed' and 'work_interfere' are replaced with 'Unknown' to avoid data loss and to
# maintain uniformity in the dataset. Handling missing values is critical for ensuring that downstream processes
# like clustering are not disrupted by incomplete data.
logging.info("Filling missing values for 'self_employed' and 'work_interfere' columns.")
data['self_employed'] = data['self_employed'].fillna('Unknown')
data['work_interfere'] = data['work_interfere'].fillna('Unknown')

# Focus on workplace-related features for clustering
# To perform a meaningful clustering analysis, we focus on features directly related to workplace mental health.
# These features provide insights into factors that might influence employee well-being and their clustering patterns.
workplace_features = ['tech_company', 'remote_work', 'benefits', 'care_options', 'wellness_program',
                      'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence',
                      'coworkers', 'supervisor']
data = data[workplace_features]
logging.info("Filtered workplace-related features for clustering.")

# Encode categorical variables
# Categorical variables are converted to numeric format using LabelEncoder to enable mathematical computations
# required for clustering algorithms. Encoding ensures that all variables are uniformly processed during analysis.
logging.info("Encoding categorical variables using LabelEncoder.")
label_encoders = {}
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Scale data for clustering
# Standardizing data with StandardScaler ensures that all features contribute equally to the clustering algorithm.
# Without scaling, features with larger numeric ranges would dominate the clustering process, biasing the results.
logging.info("Scaling data using StandardScaler.")
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Dimensionality reduction for visualization
# Principal Component Analysis (PCA) is applied to reduce data dimensions to two, enabling 2D visualization of clusters.
# While clustering operates on high-dimensional data, visualization helps interpret the results in an intuitive manner.
logging.info("Reducing data dimensions using PCA.")
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Apply KMeans clustering
# KMeans clustering divides the dataset into 4 clusters based on workplace mental health features. The choice of 4 clusters
# is based on exploratory analysis, aiming to group similar workplaces for actionable insights.
logging.info("Applying KMeans clustering with 4 clusters.")
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Visualize clusters
# The PCA-reduced data is plotted in 2D space with clusters differentiated by colors. This visualization helps in
# understanding the grouping of workplaces based on mental health-related attributes. The plot is also saved as a PNG file.
logging.info("Visualizing and saving cluster plot.")
plt.figure(figsize=(10, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=data['Cluster'], cmap='viridis', alpha=0.7)
plt.title('Workplace Mental Health Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
visualization_file = os.path.join(os.getcwd(), "cluster_visualization.png")
plt.savefig(visualization_file)
plt.show()
logging.info(f"Cluster visualization saved as {visualization_file}.")

# Analyze cluster characteristics
# The centroids of the clusters are extracted to analyze the defining characteristics of each group. This helps identify
# common traits and differences between workplace mental health clusters, offering actionable insights for interventions.
logging.info("Analyzing cluster centers.")
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=workplace_features)
cluster_centers_file = os.path.join(os.getcwd(), "cluster_centers.txt")
cluster_centers.to_csv(cluster_centers_file, index=False)
logging.info(f"Cluster centers saved as {cluster_centers_file}.")

# Print results for display
# Displaying cluster centers provides immediate feedback and allows users to interpret clustering patterns directly.
print("Cluster Centers:\n", cluster_centers)
logging.info("Clustering script execution completed.")