import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import streamlit as st
# === Загрузка ресурсов ===

def load_dataset():
    """Загружает основной обучающий датасет"""
    return pd.read_csv("data/neo_task_done1.csv")

def load_models():
    """Загружает все модели и вспомогательные объекты"""
    models = {}
    with open("knn.pkl", "rb") as f:
        models["knn"] = pickle.load(f)
    with open("gbc.pkl", "rb") as f:
        models["gbc"] = pickle.load(f)
    with open("bagg.pkl", "rb") as f:
        models["bagg"] = pickle.load(f)
    with open("stac.pkl", "rb") as f:
        models["stac"] = pickle.load(f)
    
    models["lgbm"] = lgb.Booster(model_file="lgbm.txt")
    models["fcnn"] = load_model("model_fcnn.h5")

    with open("scaler.pkl", "rb") as f:
        models["scaler"] = pickle.load(f)

    try:
        with open("label_encoder.pkl", "rb") as f:
            models["label_encoder"] = pickle.load(f)
    except FileNotFoundError:
        models["label_encoder"] = None

    return models

# === Предобработка ===

def preprocess_input_row(est_diameter_min, est_diameter_max, relative_velocity, miss_distance, absolute_magnitude, scaler):
    mean_est_diameter = (est_diameter_min + est_diameter_max) / 2
    X = np.array([[est_diameter_min, est_diameter_max, relative_velocity,
                   miss_distance, absolute_magnitude, mean_est_diameter]])
    return scaler.transform(X)

def preprocess_dataframe(df, scaler):
    if 'mean_est_diameter' not in df.columns:
        df['mean_est_diameter'] = (df['est_diameter_min'] + df['est_diameter_max']) / 2

    feature_cols = ['est_diameter_min', 'est_diameter_max', 'relative_velocity',
                    'miss_distance', 'absolute_magnitude', 'mean_est_diameter']
    return scaler.transform(df[feature_cols])

# === Предсказания ===

import numpy as np

def predict_all_models(models_dict, X_input, label_encoder=None):
    predictions = {}

    for name, model in models_dict.items():
        try:
            model_name = name.lower()

            if model_name == "lgbm":
                # LGBM booster model (not sklearn wrapper)
                proba = model.predict(X_input)
                prediction = (proba > 0.5).astype(int)
                prediction_proba = np.column_stack([1 - proba, proba])

            elif model_name == "fcnn":
                proba = model.predict(X_input, verbose=0)
                prediction = (proba > 0.5).astype(int)

                if label_encoder:
                    prediction = label_encoder.inverse_transform(prediction.reshape(-1))

                prediction_proba = np.column_stack([1 - proba, proba])

            elif hasattr(model, "predict_proba"):
                prediction_proba = model.predict_proba(X_input)
                prediction = model.predict(X_input)

            else:
                prediction = model.predict(X_input)
                prediction_proba = None

            predictions[name] = {
                "prediction": prediction.tolist(),
                "proba": prediction_proba.tolist() if prediction_proba is not None else None
            }

        except Exception as e:
            predictions[name] = {
                "prediction": None,
                "proba": None,
                "error": str(e)
            }

    return predictions

