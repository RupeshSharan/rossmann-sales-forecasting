import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_stores(df, n_clusters=3):
    features = (
        df.groupby("Store")
        .agg(
            avg_sales=("Sales", "mean"),
            sales_std=("Sales", "std"),
            promo_ratio=("Promo", "mean")
        )
        .fillna(0)
    )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    features["Cluster"] = kmeans.fit_predict(scaled)

    return features.reset_index()
