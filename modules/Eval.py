import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def hasil_eval(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "y_pred": y_pred
    }


def visual_eval(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )
    ax.set_xlabel("Actual Value")
    ax.set_ylabel("Predicted Value")
    ax.set_title("Actual vs Predicted")
    ax.grid(True)

    return fig


def koefisien_regresi(model, feature_names):

    if hasattr(model, "named_steps"):
        lr_model = model.named_steps["model"]
    else:
        lr_model = model

    coef_df = pd.DataFrame({
        "Fitur": feature_names,
        "Koefisien": lr_model.coef_
    })

    intercept = lr_model.intercept_

    return coef_df, intercept
