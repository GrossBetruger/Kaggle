import time
import pandas as pd
import matplotlib.pyplot as plt

from typing import List
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from multiprocessing import cpu_count

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

    mae_results = dict()
    timestamp = time.time()
    for n_estimators in range(50, 350, 50):

        rf_model = RandomForestRegressor(n_estimators=n_estimators,
                                         n_jobs=cpu_count(),
                                         random_state=42)
        clf = Pipeline(steps=[
            ('preprocessor', pre_proc),
            ('model', rf_model),
        ])

        cv_scores = cross_val_score(clf, X_test, y_test, cv=3, scoring='neg_mean_absolute_error')
        mae_results[n_estimators] = cv_scores.mean() * -1

    print(f"toke {time.time() - timestamp} seconds")
    plt.plot(mae_results.keys(), mae_results.values())
    plt.xlabel("# Estimators")
    plt.ylabel("MAE")
    plt.title("Housing Price Prediction (Random Forest Regressor)")
    plt.show()
