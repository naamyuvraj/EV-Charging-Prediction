from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
from preprocess import load_and_preprocess

def train():
    df_long = load_and_preprocess("../data/volume.csv")

    X = df_long[['hour', 'day_of_week', 'month', 'is_weekend']]
    y = df_long['volume_kwh']

    split_index = int(len(X) * 0.8)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("MAE:", mae)
    print("RMSE:", rmse)

    joblib.dump(model, "ev_model.pkl")
    print("Model saved as ev_model.pkl")
