import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(data):
    data = data.dropna(subset=['ARR_DEL15'])
    data['ARR_DEL15'] = data['ARR_DEL15'].astype(int)
    return data

def prepare_features(data):
    data_sample = data.sample(n=20000, random_state=42)

    X = data_sample.drop(['ARR_DEL15', 'ORIGIN', 'DEST'], axis=1)
    y = data_sample['ARR_DEL15']

    X = pd.get_dummies(X)

    return X, y