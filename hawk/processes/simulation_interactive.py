from birdy import WPSClient

train_file_path = "Emiliani1_train.csv"
test_file_path = "Emiliani1_test.csv"
target_column_name = "cyclostationary_mean_rr_4w_1"

# ----------------- WPS -----------------

wps = WPSClient("http://localhost:5002/wps", verify=False)
help(wps)

# Input some data for the causal process
resp = wps.causal(
    dataset_train=open(train_file_path),
    dataset_test=open(test_file_path),
    target_column_name=target_column_name,
    pcmci_test_choice="ParCorr",
    pcmci_max_lag="0",
    tefs_direction="both",
    tefs_use_contemporary_features="Yes",
    tefs_max_lag_features="1",
    tefs_max_lag_target="1",
)

print(resp)
resp.get()
