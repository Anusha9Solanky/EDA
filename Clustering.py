from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
clustering_features = merged_df.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'ProductID': 'count'
}).reset_index()
scaler = StandardScaler()
clustering_data = scaler.fit_transform(clustering_features.drop(columns=['CustomerID']))
num_clusters = 5  # You can experiment with this value (between 2 and 10)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(clustering_data)
clustering_features['Cluster'] = clusters
db_index = davies_bouldin_score(clustering_data, clusters)
print(f"Davies-Bouldin Index: {db_index}")
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_data = pca.fit_transform(clustering_data)

plt.figure(figsize=(10, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7)
plt.colorbar(label='Cluster')
plt.title('Customer Clusters (PCA Visualization)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid()
plt.show()
clustering_features.to_csv('Customer_Segments.csv', index=False)
print("Customer_Segments.csv created successfully!")
