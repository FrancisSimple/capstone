import joblib
import pandas as pd
import os

model_path = r"c:\Users\USER\OneDrive\Desktop\capstone\models\master\master_shelflife_model.pkl"
scaler_path = r"c:\Users\USER\OneDrive\Desktop\capstone\models\master\master_scaler.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Model loaded.")
    print("Model type:", type(model))
    if hasattr(model, 'feature_names_in_'):
        print("Feature names:", model.feature_names_in_)
    elif hasattr(model, 'n_features_in_'):
        print("Number of features:", model.n_features_in_)
kmeans_path = r"c:\Users\USER\OneDrive\Desktop\capstone\models\master\master_kmeans_4.pkl"

if os.path.exists(kmeans_path):
    kmeans = joblib.load(kmeans_path)
    print("KMeans loaded.")
    if hasattr(kmeans, 'n_features_in_'):
        print("KMeans features:", kmeans.n_features_in_)

if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("Scaler loaded.")
    if hasattr(scaler, 'feature_names_in_'):
        print("Scaler feature names:", scaler.feature_names_in_)
