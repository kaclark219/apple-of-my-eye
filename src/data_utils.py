import numpy as np
from src.features import extract_features

def load_split(split_file):
    X, y = [], []
    with open(split_file) as f:
        for line in f:
            path, label = line.strip().split("\t")
            X.append(extract_features(path))
            y.append(label)
    return np.array(X), np.array(y)