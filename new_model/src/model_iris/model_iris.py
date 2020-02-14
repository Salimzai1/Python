import numpy as np
import pandas as pd
import os
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import joblib


def read_split():
    iris = datasets.load_iris()
    df_x = pd.DataFrame(iris.data)
    df_y = pd.DataFrame(iris.target)
    return df_x, df_y


def train_iris(df_x, df_y):
    random_forest = RandomForestClassifier(
        n_estimators=150, bootstrap=True, random_state=34
    )
    model = random_forest.fit(df_x, df_y)
    return model


def get_model_path(model_dir=None):
    if model_dir is None:
        model_dir = os.path.dirname(__file__)
    model_path = os.path.join(model_dir, "iris.pkl")
    return model_path


def train_and_persist(model_dir=None):
    df_x, df_y = read_split()
    model_result = train_iris(df_x, df_y)
    model_path = get_model_path(model_dir)
    joblib.dump(model_result, model_path)


def predict(parameters, model_dir=None):
    """Returns model prediction.

    """
    model_path = get_model_path(model_dir)
    if not os.path.exists(model_path):
        train_and_persist(model_dir)
    model = joblib.load(model_path)
    X_input = pd.DataFrame([pd.Series(parameters)])
    result = model.predict(X_input)
    return int(result)
