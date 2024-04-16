from birdy import WPSClient

# ----------------- WPS -----------------

wps = WPSClient("http://localhost:5002/wps", verify=False)
help(wps)

# Input some data for the causal process
resp = wps.causal(
    dataset_train="https://raw.githubusercontent.com/climateintelligence/hawk/main/hawk/demo/Ticino_train.csv",
    dataset_test="https://raw.githubusercontent.com/climateintelligence/hawk/main/hawk/demo/Ticino_train.csv",
    target_column_name="target",
    pcmci_test_choice="ParCorr",
    pcmci_max_lag="1",
    tefs_direction="forward",
    tefs_use_contemporary_features=True,
    tefs_max_lag_features="2",
    tefs_max_lag_target="1",
)

print(resp)
resp.get()
