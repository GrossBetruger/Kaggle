import pandas as pd

from typing import List
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def read_data(file_name):
    return pd.read_csv(Path("data") / file_name)


def create_pre_processor(data: pd.DataFrame, oh_columns: List[str]) -> ColumnTransformer:
    columns = data.columns
    label_enc_columns = [c for c in columns if data[c].dtype
                   == 'object' and c not in oh_columns]
    numeric_columns = [c for c in columns if data[c].dtype
                   != 'object' and c not in oh_columns]

    numerical_transformer = SimpleImputer(strategy='mean')

    ordinal_transformer = Pipeline(steps=[
        ('ordinal_imputer', SimpleImputer(strategy='most_frequent')),
        ('label_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])

    nominal_transformer = Pipeline(steps=[
        ('nominal_imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, numeric_columns),
            ('ordinal', ordinal_transformer, label_enc_columns),
            ('nominal', nominal_transformer, one_hot_columns),
        ]
    )
    return preprocessor


if __name__ == "__main__":
    training_data = read_data("train.csv")
    one_hot_columns = ['Street', 'Alley', 'LandContour', 'LandSlope', 'RoofMatl', 'RoofStyle',
                       'HouseStyle', 'Heating', 'Exterior2nd', 'Exterior1st']
    # X, y = pre_process_data(training_data, one_hot_columns)
    X = training_data.drop("SalePrice", axis=1)
    y = training_data.SalePrice
    pre_proc = create_pre_processor(X, oh_columns=one_hot_columns)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33,
                                                        random_state=42)
    rf_model = RandomForestRegressor(random_state=42)
    clf = Pipeline(steps=[
        ('preprocessor', pre_proc),
        ('model', rf_model),
    ])

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("MAE:", mean_absolute_error(pred, y_test))
    print("MSE:", mean_squared_error(pred, y_test))
