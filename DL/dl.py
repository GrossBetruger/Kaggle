from enum import auto, Enum
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing


class ModelType(Enum):
    Deep = auto()
    Single = auto()


def normalize_zero_one(data: DataFrame):
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    return pd.DataFrame(data_scaled)


def prepare_cereal_data(model_type: ModelType) -> Tuple[DataFrame, Series]:
    cereal_data = pd.read_csv(Path("data") / "cereal.csv")
    features = ['protein', 'fat', 'sodium', 'fiber',
                'carbo', 'sugars', 'potass', 'vitamins', 'shelf', 'weight', 'cups',
                'rating']
    if model_type is ModelType.Single:
        features = ['fat', 'sugars']
    X = cereal_data[features].copy()

    # normalized values of X between [0, 1]
    # deep learning models tend to better with normalized input
    X = normalize_zero_one(X)

    y = cereal_data['calories']
    return X, y


def get_single_layer_model(num_features: int) -> keras.Sequential:
    model = keras.Sequential([
        layers.Dense(units=1, input_shape=[num_features])
    ])

    model.compile(optimizer="SGD", loss="mse", metrics=["mae"])
    return model


def get_deep_model(num_features: int) -> keras.Sequential:
    model = keras.Sequential([

        # hidden ReLU layers
        layers.Dense(units=4, activation='relu', input_shape=[num_features]),
        layers.Dense(units=3, activation='elu'),
        layers.Dense(units=2, activation='relu'),

        # linear output layer (no activation)
        layers.Dense(units=1)
    ])

    model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    return model


def cereal_model_single_layer_main():
    model_t = ModelType.Deep

    X, y = prepare_cereal_data(model_t)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    if model_t is ModelType.Single:
        model = get_single_layer_model(len(X.columns))
    elif model_t is ModelType.Deep:
        model = get_deep_model(len(X.columns))
    else:
        raise Exception("Undefined Model Type")

    num_epochs = 3000
    history = model.fit(X_train, y_train, epochs=num_epochs)
    history = pd.DataFrame(history.history)
    history['loss'].plot()
    plt.show()

    for i in range(5):
        x = X_test.iloc[[i]]
        y = list(y_test)[i]
        print(f"model prediction: {model.predict([x])[0][0]}, true value: {y}")
        print()


if __name__ == '__main__':
    cereal_model_single_layer_main()
