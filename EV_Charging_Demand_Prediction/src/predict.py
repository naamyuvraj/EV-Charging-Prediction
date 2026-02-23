from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def make_predictions(model, X_test, y_test):
    """
    Generate predictions using data

    parameters:
        model,X_test,y_test

    predict:
        mae,rmse
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return predictions, mae, rmse