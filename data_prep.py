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
