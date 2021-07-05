import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def read_data(file_name):
    return pd.read_csv(Path("data") / file_name)


def pre_process_data(data):
    le = LabelEncoder()
    columns = data.columns
    non_numeric = [c for c in columns if data[c].dtype
                   not in ['int64', 'float64']]
    data[non_numeric] = data[non_numeric].apply(le.fit_transform)
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data))
    data.columns = columns
    y = data.SalePrice
    X = data[set(data.columns) - {"SalePrice"}]
    return X, y


if __name__ == "__main__":
    training_data = read_data("train.csv")
    X, y = pre_process_data(training_data)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33,
                                                        random_state = 42)
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    pred = rf_model.predict(X_test)
    print("MAE:", mean_absolute_error(pred, y_test))
    print("MSE:", mean_squared_error(pred, y_test))
