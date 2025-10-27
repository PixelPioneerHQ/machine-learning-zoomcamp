import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"
columns = ["engine_displacement", "horsepower", "vehicle_weight", "model_year", "fuel_efficiency_mpg"]
features = columns[:-1]
target = "fuel_efficiency_mpg"

df = pd.read_csv(url, usecols=columns)

r_values = [0, 0.01, 0.1, 1, 5, 10, 100]

def shuffle_split(dataframe, seed=42):
    np.random.seed(seed)
    idx = np.arange(len(dataframe))
    np.random.shuffle(idx)
    df_shuffled = dataframe.iloc[idx].reset_index(drop=True)
    n = len(df_shuffled)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    df_train = df_shuffled.iloc[:n_train].reset_index(drop=True)
    df_val = df_shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test = df_shuffled.iloc[n_train + n_val:].reset_index(drop=True)
    return df_train, df_val, df_test


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

rmse_by_r = {}
for r in r_values:
    df_train, df_val, _ = shuffle_split(df, seed=42)
    X_train = df_train[features].fillna({"horsepower": 0}).values
    y_train = df_train[target].values
    X_val = df_val[features].fillna({"horsepower": 0}).values
    y_val = df_val[target].values
    model = Ridge(alpha=r)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse_by_r[r] = rmse(y_val, preds)

print(rmse_by_r)
