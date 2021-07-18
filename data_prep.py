import pandas as pd
import numpy as np


def fetch_data(filepath='data/Foreign_Exchange_Rates.csv'):
    df = pd.read_csv('data/Foreign_Exchange_Rates.csv', index_col=0, na_values=['ND'])
    df = df[['Time Serie', 'EURO AREA - EURO/US$']]
    df.columns = ['day', 'EURO']
    df['day'] = pd.to_datetime(df['day'])
    return df.set_index('day')

def scale_and_interpolate(df):
    work_df = (df - df.min()) / (df.max() - df.min())
    work_df = work_df.join(pd.DataFrame(index=pd.date_range('2000-01-03', '2019-12-31')), how='outer')
    work_df.interpolate(inplace=True)
    work_df.index.rename('day', inplace=True)
    return work_df

def set_target(df, window=2):
    df = df.copy()
    def rolling_func(x):
        if len(x) < window:
            return -1
        return x[-1] < x[0]


    df['target'] = (
        df
        .sort_values(by='day', ascending=False)[['EURO']]
        .rolling(f'{window}d', closed='both')
        .agg(rolling_func)
        .sort_values(by='day', ascending=True)
    )
    return df[df['target'] != -1]


def expand_columns(df, col_names, steps=1):
    """
    Make backstep features for each column in `col_names`

    :param steps: The number of backsteps to go
    """
    df = df.copy()
    for col_name in col_names:
        for i in range(1, steps+1):
            df[f'{col_name}_{i}'] = df[col_name].shift(i)
    
    # Drop the first few columns that don't have all the back steps
    return df.iloc[steps:]
