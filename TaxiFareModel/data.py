import pandas as pd

CSV_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"
# CSV_PATH = "../raw_data/train_10k.csv"


def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    print('fetching data...')
    df = pd.read_csv(CSV_PATH, nrows=nrows)
    df = clean_data(df)
    return get_features_targets(df)



def clean_data(df, test=False):
    print('cleaning data...')
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    print('cleaning data done...')
    return df

def get_features_targets(df):
    y = df["fare_amount"]
    x = df.drop("fare_amount", axis=1)
    return x, y


if __name__ == '__main__':
    df = get_data()
