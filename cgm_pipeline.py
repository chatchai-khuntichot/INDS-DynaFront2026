# ==============================================================================
# INDS DATA PIPELINE: HEALTHCARE CONTINUOUS GLUCOSE MONITORING (CGM)
# AUTHOR: S.Wattanasuphinyakit Author(s)
# STRICTLY ALIGNED WITH PAPER: No Temporal Leakage, |E|=12 Environments
# ==============================================================================

import xml.etree.ElementTree as ET
import numpy as np
import torch


def parse_ohio_xml_to_envs(file_path, seq_length=100, n_envs=12):

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        raise FileNotFoundError(f"Cannot read XML file at {file_path}. Error: {e}")


    glucose_data = []
    for event in root.findall('.//glucose_level/event'):
        val = event.get('value')
        if val is not None:
            glucose_data.append(float(val))

    glucose_data = np.array(glucose_data)
    total_len = len(glucose_data)

    env_size = total_len // n_envs
    if env_size <= seq_length:
        raise ValueError(f"Dataset too short to split into {n_envs} envs with window {seq_length}")

    X_envs, Y_envs = [], []

    for e in range(n_envs):
        start_idx = e * env_size
        end_idx = (e + 1) * env_size if e < n_envs - 1 else total_len
        env_data = glucose_data[start_idx:end_idx]

        X_e, Y_e = [], []
        for i in range(len(env_data) - seq_length):
            X_e.append(env_data[i: i + seq_length])
            Y_e.append(env_data[i + 1: i + seq_length + 1])

        X_envs.append(torch.tensor(np.array(X_e), dtype=torch.float32))
        Y_envs.append(torch.tensor(np.array(Y_e), dtype=torch.float32))

    return X_envs, Y_envs


def parse_ohio_xml_to_tensor(file_path, seq_length=100):

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        raise FileNotFoundError(f"Cannot read XML file at {file_path}. Error: {e}")

    glucose_data = []
    for event in root.findall('.//glucose_level/event'):
        val = event.get('value')
        if val is not None:
            glucose_data.append(float(val))

    glucose_data = np.array(glucose_data)
    X, Y = [], []

    for i in range(len(glucose_data) - seq_length):
        X.append(glucose_data[i: i + seq_length])
        Y.append(glucose_data[i + 1: i + seq_length + 1])

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y), dtype=torch.float32)

    return X_tensor, Y_tensor