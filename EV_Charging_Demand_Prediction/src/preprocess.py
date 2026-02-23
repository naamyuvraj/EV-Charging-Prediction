import pandas as pd

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])

    df_long = df.melt(
        id_vars=['time'],
        var_name='TAZID',
        value_name='volume_kwh'
    )
    df_long['hour'] = df_long['time'].dt.hour
    df_long['day_of_week'] = df_long['time'].dt.dayofweek
    df_long['month'] = df_long['time'].dt.month
    df_long['is_weekend'] = (df_long['day_of_week'] >= 5).astype(int)

    return df_long