import numpy as np
import pandas as pd
from birdy import WPSClient

# from keras import models


np.random.seed(0)
n = 1000  # number of samples
m = 15  # number of features

data = {}
for i in range(1, m + 1):
    data[f"x{i}"] = np.random.normal(size=n)

data["y"] = sum(data.values()) + np.random.normal(size=n)

data = pd.DataFrame(data)

n_test = int(0.20 * n)
n_train = n - n_test
data_test = data[n_train:]
data = data[:n_train]

data.head()

target_name = "y"

url = "http://localhost:5000/wps"
wps = WPSClient(url, verify=False)
help(wps)

resp = wps.causal()
print(resp)
resp.get()
