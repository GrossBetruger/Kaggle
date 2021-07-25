import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from enum import auto, Enum
from pathlib import Path
from typing import Tuple
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing


tf.random.set_seed(42)


class ModelType(Enum):
    Deep = auto()
    Single = auto()


def normalize_zero_one(data: DataFrame):
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    return pd.DataFrame(data_scaled)


def prepare_cereal_data(model_type: ModelType, encode_categorical=False) -> Tuple[DataFrame, Series]:
    cereal_data = pd.read_csv(Path("data") / "cereal.csv")
    target = 'calories'
    # features = ['protein', 'fat', 'sodium', 'fiber',
    #             'carbo', 'sugars', 'potass', 'vitamins', 'shelf', 'weight', 'cups',
    #             'rating']

    # drop target leak column
    cereal_data.drop('name', axis=1, inplace=True)

    # handle categorical columns
    categorical_cols = ['mfr', 'type']
    if encode_categorical is False:  # dropping seems more successful for this dataset
        cereal_data.drop(categorical_cols, axis=1, inplace=True)
    else:
        # one hot categoricals
        cereal_data = pd.get_dummies(cereal_data, columns=categorical_cols)

    if model_type is ModelType.Single:
        features = ['fat', 'sugars']
    y = cereal_data[target]

    # drop target feature
    X = cereal_data.drop(target, axis=1)

    # normalized values of X between [0, 1]
    # deep learning models tend to better with normalized input
    X = normalize_zero_one(X)

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

    # opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
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

    num_epochs = 24000
    history = model.fit(X_train, y_train, epochs=num_epochs)
    history = pd.DataFrame(history.history)
    history['loss'].plot()
    plt.show()

    for i in range(20):
        x = X_test.iloc[[i]]
        y = list(y_test)[i]
        print(f"model prediction: {model.predict([x])[0][0]}, true value: {y}")
        print()


if __name__ == '__main__':
    cereal_model_single_layer_main()
