import os
from pathlib import Path

import pandas as pd

from dataprep import DataPrep

CAT = ['type', 'operation', 'k_symbol', 'bank', 'account']
LOG = ['amount']
MIXED = {'k_symbol': [0], 'bank': [0], 'account': [0]}
INTEGER = []
PROBLEM = {"Classification": 'type'}


def get_cz_bank_data(acc_id=None):
    csv_path = os.path.join('data', 'clean_trans.csv')
    df = pd.read_csv(csv_path)
    df.date = pd.to_datetime(df.date)
    df = df.set_index('date')
    # for i in df['account_id'].unique():
    #     sing_acc = df.loc[df['account_id'] == i]
    #     if 200 < len(sing_acc) < 300:
    #         print(i, len(sing_acc))
    data_raw = df.loc[df['account_id'] == acc_id]
    data_raw = data_raw.drop(columns=['account_id', 'trans_id', 'balance'])
    data_prep = DataPrep(data_raw, categorical=CAT,
                         log=LOG,
                         mixed=MIXED,
                         integer=INTEGER,
                         type=PROBLEM,
                         test_ratio=0.2)
    data_onthot = pd.get_dummies(data_prep.df, columns=CAT, drop_first=True).sort_index()
    df_out = data_onthot.groupby(pd.Grouper(freq='14D')).sum()
    return df_out.to_numpy()
