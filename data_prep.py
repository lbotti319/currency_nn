import pandas as pd
import numpy as np


def fetch_data(filepath="data/Foreign_Exchange_Rates.csv", num_currencies=1):
    df = pd.read_csv("data/Foreign_Exchange_Rates.csv", index_col=0, na_values=["ND"])
    if num_currencies == 1:
        df = df[["Time Serie", "EURO AREA - EURO/US$"]]
        df.columns = ["day", "EURO"]
    
    elif num_currencies == 2:
        df = df[["Time Serie", "EURO AREA - EURO/US$", "UNITED KINGDOM - UNITED KINGDOM POUND/US$"]]
        df.columns = ["day", "EURO", "POUND"]
    
    df["day"] = pd.to_datetime(df["day"])
    return df.set_index("day")


def interpolate(df):
    work_df = df.copy()
    work_df = work_df.join(
        pd.DataFrame(index=pd.date_range("2000-01-03", "2019-12-31")), how="outer"
    )
    work_df.interpolate(inplace=True)
    work_df.index.rename("day", inplace=True)
    return work_df


def set_target(df, window=1, target_col="EURO"):
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
    """
    df = df.copy()
    for col_name in col_names:
        for i in range(1, steps + 1):
            df[f"{col_name}_{i}"] = df[col_name].shift(i)

    # Drop the first few columns that don't have all the back steps
    return df.iloc[steps:]

def backstep_columns(df, columns=['EURO'], steps=3):
    N = df.shape[0] - steps
    # Increment steps by 1 to account for the current day
    steps = steps +1
    output = np.zeros((N, steps, len(columns)))
    for i, c in enumerate(columns):
        for j in range(steps):
            # Fill in backwards, so the oldest is last
            output[:,steps-j-1,i] = df[c].shift(j)[steps-1:]
    return output

def train_test_split(features, targets, percentile=None, test_window=None):
    """
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
