import pickle
import pandas as pd

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def save_columns(columns, path):
    pd.Series(columns).to_csv(path, index=False)