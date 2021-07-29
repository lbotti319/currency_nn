import pandas as pd
import numpy as np
import torch


def fetch_data(augment=True):
    """
    Returns EURO price by day. If augment is true, it also has the EMU data as feature columns
    """
    df = pd.read_csv("data/Foreign_Exchange_Rates.csv", index_col=0, na_values=["ND"])
    df = df[["Time Serie", "EURO AREA - EURO/US$"]]
    df.columns = ["day", "EURO"]
    
    df["day"] = pd.to_datetime(df["day"])
    df.set_index("day", inplace=True)
    df = interpolate(df)
    if augment:
        emu = _get_emu()
        df = df.join(emu, how='left').interpolate().fillna(0.0)
    return df

def _get_emu():
    """
    Retrieves the EMU data and cleans it. Drops columns with lots of NaN values.
    """
    raw_df = pd.read_csv('data/irt_lt_mcby_d.tsv.gz', sep="\t", na_values=[": z"])
    raw_df.head()
    df = raw_df.set_index(raw_df['int_rt,geo\\time'].str.split(",").apply(lambda x: x[1])).drop('int_rt,geo\\time', axis=1)
    # df.index.rename('date', inplace=True)
    df = df.transpose()
    df.reset_index(inplace=True)
    df['date'] = df['index'].str.strip()
    df.drop('index', axis=1, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format="%YM%mD%d")
    df.set_index('date', inplace=True)
    df.columns.rename('country_code', inplace=True)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # Drop columns with lots of nans, or that are hard to interpolate
    return df.drop(['EE', 'HR', 'RO', 'BG', 'SI'], axis=1)


def interpolate(df):
    """
    Ensures every day from in the range is present in the dataset. Missing values are interpolated
    """
    work_df = df.copy()
    work_df = work_df.join(
        pd.DataFrame(index=pd.date_range("2000-01-03", "2019-12-31")), how="outer"
    )
    work_df.interpolate(inplace=True)
    work_df.index.rename("day", inplace=True)
    return work_df


def set_target(df, window=1, target_col="EURO"):
    """
    Creates a target column looking forward "window" number of days
    Drops samples without targets
    """
    df = df.copy()

    df["target"] = df[target_col].shift(-1 * window)
    df = df[df["target"].notna()]
    df["target"] = (df["target"] > df[target_col]).astype(np.int64)
    return df


def set_target_regression(df, window=1, target_col="EURO"):
    df = df.copy()

    df["target"] = df[target_col].shift(-1 * window)
    return df[df["target"].notna()]


def expand_columns(df, col_names, steps=1):
    """
    Make backstep features for each column in `col_names`

    :param steps: The number of backsteps to go
    
    Returns a two dimensional dataframe (samples, features)
    """
    df = df.copy()
    for col_name in col_names:
        for i in range(1, steps + 1):
            df[f"{col_name}_{i}"] = df[col_name].shift(i)

    # Drop the first few columns that don't have all the back steps
    return df.iloc[steps:]

def backstep_columns(df, steps=3):
    """
    Returns a three dimensional dataframe (samples, steps, features)
    """
    N = df.shape[0] - steps
    # Increment steps by 1 to account for the current day
    steps = steps +1
    output = np.zeros((N, steps, len(df.columns)))
    for i, c in enumerate(df.columns):
        for j in range(steps):
            # Fill in backwards, so the oldest is last
            output[:,steps-j-1,i] = df[c].shift(j)[steps-1:]
    return output

def train_test_split(features, targets, percentile=None, test_window=None):
    """
    Splits the data into train and test sets. If percentile is provided, it splits the whole series.
    If test_window is provided it returns all samples before the window as the train set, and the window as the test set.
    Samples after the window are ignored. Used for backtesting
    
    
    :param percentile: float or None
    :param test_window: tuple or None 
    """
    if percentile is not None:
        cutoff = int(features.shape[0]*percentile)
        X_train, X_test = features[:cutoff], features[cutoff:]
        y_train, y_test = targets[:cutoff], targets[cutoff:]
    elif test_window is not None:
        start = test_window[0]
        end = test_window[1]
        X_train, X_test = features[:start], features[start:end]
        y_train, y_test = targets[:start], targets[start:end]
    return X_train, X_test, y_train, y_test
