"""
This is a more 'exploratory' module than the others, and does not inherit from the abstract Module class.
It is used to build a logistic regression model that predicts the probability of a pixel being a break point.
The model is built using the train_inv.csv file, which contains the points that will be used for training.
The features are the NCI values, the delta values and the raw NDVI value.
The model is built using the sklearn library.

It should only be run again if the train_inv.csv file is updated, i.e., if a new selection of points are used for
training. The results are then updated in the utils.py file, in the coefficients_log_reg dictionary.
"""

import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config.config import Config
from config.io_config import IOConfig
from io_manager.io_manager import IO


def build_logistic_regression_model():
    """
    Builds a logistic regression model that predicts the probability of a pixel being a break point.
    """
    # Load the dataset
    df = pd.read_csv(f'{io.config.base_local_dir}/operation_records/train_features.csv')

    # Split the dataset into features and target
    X = df.drop(columns=['y', 'i', 'j', 'year', 'tile', 'date', 'raw'], axis=1)
    y = df['y']

    # Print the shape of the features and target
    print(f'Features shape: {X.shape}')
    print(f'Target shape: {y.shape}')

    # Feature engineering
    def feature_engineering(X):
        # Any inf value takes the mean of the column
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())

        return X

    X = feature_engineering(X)

    # Create a logistic regression model
    model = LogisticRegression(max_iter=300)

    # Fit the model on the training data
    model.fit(X, y)

    # Evaluate the model with k-fold cross-validation
    scores = {
        'accuracy': cross_val_score(model, X, y, cv=5, scoring='accuracy'),
        'precision': cross_val_score(model, X, y, cv=5, scoring='precision'),
        'recall': cross_val_score(model, X, y, cv=5, scoring='recall'),
        'f1': cross_val_score(model, X, y, cv=5, scoring='f1')
    }

    print(f'Scores: {scores} \n'
          f'Accuracy: {scores["accuracy"].mean()} \n'
          f'Precision: {scores["precision"].mean()} \n'
          f'Recall: {scores["recall"].mean()} \n'
          f'F1: {scores["f1"].mean()} \n')

    # Print the parameters of the model
    coef_dict = {'intercept': model.intercept_[0]}
    for i, col in enumerate(X.columns):
        coef_dict[col] = model.coef_[0][i]

    print(f'Coefficients: {coef_dict}')

    # Print model results
    y_pred = [random.randint(0, 1) for _ in range(len(y))]
    print(f'{y_pred}')
    print(f'Random classifier: ')
    print(f'  Accuracy: {accuracy_score(y, y_pred)}')
    print(f'  Precision: {precision_score(y, y_pred)}')
    print(f'  Recall: {recall_score(y, y_pred)}')
    print(f'  F1: {f1_score(y, y_pred)}')

    return model, X, y, coef_dict


def check_predictions(model, X, y, coef_dict):
    """
    This is a check to see if the predictions of the model are the same as the predictions built manually.
    It is important because in the change detection module, we won't use the sklearn model object, but just use
    the coefficients to build the predictions manually, which saves time.
    """
    # Predict on first 10 rows
    y_pred = model.predict_proba(X[:10])
    y_pred = [x[1] for x in y_pred]

    # Now build the same predictions manually
    y_pred_manual = []
    for i, row in X[:10].iterrows():
        point = coef_dict['intercept']
        for col in row.index:
            point += coef_dict[col] * row[col]
        point = 1 / (1 + np.exp(-point))
        y_pred_manual.append(point)

    print(f'Predictions: {y_pred}')
    print(f'Predictions manual: {y_pred_manual}')

    # Compare differences:
    print(f'Predictions diff: {np.array(y_pred) - np.array(y_pred_manual)}')


if __name__ == "__main__":
    config = Config()
    io_config = IOConfig()
    io = IO(io_config)

    model, X, y, coef_dict = build_logistic_regression_model()
    check_predictions(model, X, y, coef_dict)
