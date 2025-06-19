from typing import Union, Dict

import pandas as pd
import torch


def predict(model, preprocessor, data_path: str):
    print("Formatting prediction data...")
    df = pd.read_csv(data_path)

    # Use saved feature cols to initialise dataframe
    try:
        num_features = preprocessor.transformers_[0][2]
        cat_features = preprocessor.transformers_[1][2]
        feature_cols = num_features + cat_features
        X = df[feature_cols]
    except Exception as e:
        print(f"Error preparing data for preprocessing: {e}")
        print("Please ensure the input data contains all required columns:", feature_cols)
        return None

    # Get name and score cols
    biscuit_names = df['Biscuit Name']
    target = df['Score']

    # Preprocessing data
    X= preprocessor.transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    print("Making predictions...")
    with torch.no_grad():
        result = model(X_tensor)

    # Reformatting result
    result = result.detach().numpy().flatten()

    return biscuit_names, result, target
