# modules/modelling.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def modelling(df_log):

    # =============================
    # 1. PISAH FITUR & TARGET
    # =============================
    X = df_log.drop(columns=['Production'])
    y = df_log["Production"]

    # =============================
    # 2. SPLIT DATA
    # =============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # =============================
    # 3. PIPELINE (IMPUTER + MODEL)
    # =============================
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", LinearRegression())
    ])

    # =============================
    # 4. TRAINING
    # =============================
    pipeline.fit(X_train, y_train)

    return pipeline, X_train, X_test, y_train, y_test
