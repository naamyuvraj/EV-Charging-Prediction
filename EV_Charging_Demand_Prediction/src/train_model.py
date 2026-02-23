from sklearn.linear_model import LinearRegression

def train_model(df_long):
    X = df_long[['hour', 'day_of_week', 'month', 'is_weekend']]
    y = df_long['volume_kwh']

    split_index = int(len(X) * 0.8)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test