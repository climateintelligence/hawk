import numpy as np
import pandas as pd
from birdy import WPSClient

# ----------- Generate some data -----------
np.random.seed(0)
n = 1000  # number of samples
m = 15  # number of features

data = {}
for i in range(1, m + 1):
    data[f"x{i}"] = np.random.normal(size=n)

target_name = "target"
data[target_name] = sum(data.values()) + np.random.normal(size=n)

data = pd.DataFrame(data)

n_test = int(0.20 * n)
n_train = n - n_test
data_test = data[n_train:]
data = data[:n_train]

data.head()


train_file_path = "./train_dataset.csv"
test_file_path = "./test_dataset.csv"
data.to_csv(train_file_path, index=False)
data_test.to_csv(test_file_path, index=False)

# ----------------- WPS -----------------

wps = WPSClient("http://localhost:5000/wps", verify=False)
help(wps)

# Input some data for the causal process
resp = wps.causal(
    dataset_train=train_file_path,
    dataset_test=test_file_path,
    target_column_name=target_name,
    pcmci_test_choice="ParCorr",
    pcmci_max_lag="1",
    tefs_direction="both",
    tefs_use_contemporary_features="Yes",
    tefs_max_lag_features="1",
    tefs_max_lag_target="1",
)

print(resp)
resp.get()
