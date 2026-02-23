import pandas as pd

# 1️⃣ Load data
df = pd.read_csv("data/volume.csv")
df['time'] = pd.to_datetime(df['time'])

# 2️⃣ Convert wide → long
df_long = df.melt(
    id_vars=['time'],
    var_name='TAZID',
    value_name='volume_kwh'
)

# 3️⃣ Add time features
df_long['hour'] = df_long['time'].dt.hour
df_long['day_of_week'] = df_long['time'].dt.dayofweek
df_long['month'] = df_long['time'].dt.month
df_long['is_weekend'] = (df_long['day_of_week'] >= 5).astype(int)

print(df_long.head())
print("\nMissing values:\n", df_long.isnull().sum())