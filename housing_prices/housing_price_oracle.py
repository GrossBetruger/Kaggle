import pandas as pd

from typing import List
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def read_data(file_name):
    return pd.read_csv(Path("data") / file_name)


def pre_process_data(data: pd.DataFrame, oh_columns: List[str]):
    columns = data.columns
    label_enc_columns = [c for c in columns if data[c].dtype
                   == 'object' and c not in oh_columns]
    numeric_columns = [c for c in columns if data[c].dtype
                   != 'object' and c not in oh_columns]

    # separate into 3 frames (each requires different encoding)
    oh_data = data.drop(set(columns) - set(oh_columns), axis=1)
    numeric_data = data.drop(set(label_enc_columns) | set(oh_columns), axis=1)
    label_enc_data = data.drop(set(numeric_columns) | set(oh_columns), axis=1)

    # Label Encode
    le = LabelEncoder()
    label_enc_data = label_enc_data.apply(le.fit_transform)

    # One-Hot Encode
    oh = OneHotEncoder(sparse=False, handle_unknown='ignore')
    oh.fit(oh_data[oh_columns])
    expanded_one_hot_columns = oh.get_feature_names(oh_columns)
    transformed_oh_data = pd.DataFrame(oh.transform(oh_data[oh_columns]), columns=expanded_one_hot_columns)

    # Concat frames back together
    data = pd.concat([numeric_data, transformed_oh_data, label_enc_data], axis=1)

    # Impute missing data
    imputer = SimpleImputer(strategy='mean')
    columns = data.columns # imputer loses column names
    data = pd.DataFrame(imputer.fit_transform(data))
    data.columns = columns

    # Separate target data
    y = data.SalePrice
    X = data[set(data.columns) - {"SalePrice"}]

    return X, y


if __name__ == "__main__":
    training_data = read_data("train.csv")
    one_hot_columns = ['Street', 'Alley', 'LandContour', 'LandSlope', 'RoofMatl', 'RoofStyle',
                       'HouseStyle', 'Heating', 'Exterior2nd', 'Exterior1st']
    X, y = pre_process_data(training_data, one_hot_columns)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33,
                                                        random_state = 42)
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    pred = rf_model.predict(X_test)
    print("MAE:", mean_absolute_error(pred, y_test))
    print("MSE:", mean_squared_error(pred, y_test))
