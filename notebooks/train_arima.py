# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: pred-ops-os
#     language: python
#     name: pred-ops-os
# ---

# %%
from datetime import datetime

import numpy as np
from common import run2seq
from constants import DATA_ROOT
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

# %%
INPUT_SAMPLES = 20
PREDICTION_OFFSET_MIN = 15
DERIVATIVE = 1

limit = 2880

# %%
scaler = MinMaxScaler()
scaler.fit(np.array([[0], [400]]))

data = run2seq(DATA_ROOT / "train_super_steep_behavior.csv").reshape(-1)
data = data[:limit]

data_scaled = scaler.transform(data.reshape(-1, 1)).reshape(-1)
data_scaled

# %%
train_start = datetime.now()
model = ARIMA(data_scaled, order=(INPUT_SAMPLES, DERIVATIVE, 0)).fit()
train_end = datetime.now()

# %%
train_end - train_start

# %%
current_date = datetime.today().strftime("%Y-%m-%d")
dump_filename = f"arima-{INPUT_SAMPLES:02}-{DERIVATIVE}-0_sum_{current_date}.sm"
print(dump_filename)
model.save(dump_filename)

# %%
