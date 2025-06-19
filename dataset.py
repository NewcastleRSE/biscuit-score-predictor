import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocessing(path: str):
    print("Data Preprocessing...")
    df = pd.read_csv(path, sep = ',')

    # Target col and list of cols to be excluded
    target_col = ['Score']
    exclude_cols = ['Biscuit Name', 'Website for nutritional info per 100g', 'Score']

    # Get all feature columns
    all_cols = df.columns.tolist()
    feature_cols = [col for col in all_cols if col not in exclude_cols]

    # Separate columns with numerical and categorical features
    numerical_features = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    # print(numerical_features)
    # print(categorical_features)

    # Setup StandardScaler for numerical data
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Setup OneHotEncoder for categorical data:
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Setup preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Keep other columns
    )

    # Apply transformer to dataset
    X = df.drop(columns=exclude_cols)
    X = preprocessor.fit_transform(X)
    y = df[target_col]
    # print(X)
    # print(y)

    # Convert to Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    return X_tensor, y_tensor, preprocessor
