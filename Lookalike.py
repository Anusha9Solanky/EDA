from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pandas as pd
merged_df = transactions_df.merge(customers_df, on='CustomerID', how='left')
merged_df = merged_df.merge(products_df, on='ProductID', how='left')
customer_features = merged_df.groupby('CustomerID').agg({
    'TotalValue': 'sum',  
    'Quantity': 'sum',   
    'ProductID': 'count', 
}).reset_index()
customer_features = customer_features.rename(columns={'ProductID': 'TransactionCount'})
scaler = StandardScaler()
features_scaled = scaler.fit_transform(customer_features.drop(columns=['CustomerID']))
similarity_matrix = cosine_similarity(features_scaled)
lookalike_results = {}
for i in range(20):
    customer_id = customer_features['CustomerID'].iloc[i]
    similarity_scores = list(enumerate(similarity_matrix[i]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_3 = [(customer_features['CustomerID'][idx], score) for idx, score in similarity_scores[1:4]]
    lookalike_results[customer_id] = top_3
lookalike_df = pd.DataFrame([{
    'CustomerID': cust_id,
    'Lookalikes': str([(x[0], round(x[1], 2)) for x in lookalike_results[cust_id]])
} for cust_id in lookalike_results])

lookalike_df.to_csv('Lookalike.csv', index=False)
print("Lookalike.csv created successfully!")
