import numpy as np
import pandas as pd
from birdy import WPSClient
#from keras import models


url = "http://localhost:5000/wps"
wps = WPSClient(url, verify=False)
help(wps)


resp = wps.hello(name="Pluto")
print(resp)
resp.get()


resp = wps.cyclone(start_day="2019-01-04", end_day="2019-01-06", area="Sindian")
print(resp)
resp.get()
