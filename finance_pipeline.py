# ==============================================================================
# INDS DATA PIPELINE: FINANCIAL (XAUUSD HOURLY CLOSE)
# AUTHOR: S.Wattanasuphinyakit Author(s)
# STRICTLY ALIGNED WITH PAPER: No Artificial Truncation, |E|=8 ID Environments
# ==============================================================================

import pandas as pd
import numpy as np
import torch


def prepare_real_xau_ood_datasets(file_path="data/XAUUSD_H1_202305282200_202605280000.csv", window_size=100, n_envs=8):
    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        raise FileNotFoundError(f"Cannot read Financial CSV file at {file_path}. Error: {e}")

    df['Datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])

    id_mask = df['Datetime'] <= '2024-12-31 23:00:00'
    ood_mask = df['Datetime'] >= '2025-01-01 00:00:00'

    id_data = df.loc[id_mask, '<CLOSE>'].values
    ood_data = df.loc[ood_mask, '<CLOSE>'].values


    fin_min = id_data.min()
    fin_max = id_data.max()


    def create_windows(data):
        X, Y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i: i + window_size])
            Y.append(data[i + 1: i + window_size + 1])
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)


    X_ood, Y_ood = create_windows(ood_data)

    env_size = len(id_data) // n_envs
    X_id_envs, Y_id_envs = [], []

    for e in range(n_envs):
        start_idx = e * env_size
        end_idx = (e + 1) * env_size if e < n_envs - 1 else len(id_data)
        env_data = id_data[start_idx:end_idx]

        X_e, Y_e = create_windows(env_data)
        X_id_envs.append(X_e)
        Y_id_envs.append(Y_e)

    return X_id_envs, Y_id_envs, X_ood, Y_ood, fin_min, fin_max