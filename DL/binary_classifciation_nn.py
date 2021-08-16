import pandas as pd

from pathlib import Path

import tensorflow.python.keras.callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

model: keras.Sequential = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=(34,)),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

early_stopping = tensorflow.keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

lonosphere_data: pd.DataFrame = pd.read_csv(Path('data') / 'ionosphere.data')

if __name__ == '__main__':
    le = LabelEncoder()
    lonosphere_data.g = le.fit_transform(lonosphere_data.g)
    # print(le.inverse_transform(lonosphere_data.g))
    y = lonosphere_data.g
    lonosphere_data.drop(['g'], axis=1, inplace=True)
    X = lonosphere_data
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=512,
        epochs=20000,
        callbacks=[early_stopping],
        verbose=1
    )