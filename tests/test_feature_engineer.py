import pandas as pd
import tempfile
import os
from src.feature_engineer import SARIMAprep, TLSTMprep, RandomForestPrep

def test_sarima_prep_splits_and_imputation():
    # create a sample DataFrame with missing values
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=48, freq="H"),
        "value": [i if i % 5 != 0 else None for i in range(48)]
    })

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f.name, index=False)
        sarima = SARIMAprep(
            data_path=f.name,
            timestamp_col="timestamp",
            split_timestamp="2023-01-01 23:00:00",
            value_col="value",
            method="ffill"
        )
        train, test = sarima.get_train_test()
        assert not train["value"].isna().any(), "Train set has missing values after imputation"
        assert not test["value"].isna().any(), "Test set has missing values after imputation"
    os.unlink(f.name)

def test_tlstm_prep_dataset_shapes():
    # mock data
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=50, freq="H"),
        "reading_value_hourly_mean": range(50)
    })

    # save to 2 fake CSVs
    with tempfile.TemporaryDirectory() as tmpdir:
        df.to_csv(os.path.join(tmpdir, "sensor1.csv"), index=False)
        df.to_csv(os.path.join(tmpdir, "sensor2.csv"), index=False)

        prep = TLSTMprep(
            datapath=tmpdir,
            lookback=5,
            horizon=1,
            batch_size=8,
            target_col="sensor1",
            split_date="2023-01-02"
        )

        x, t, y = next(iter(prep.train_loader))
        assert x.shape[-1] > 0, "X has no features"
        assert t.shape[-1] == 1, "Elapsed time tensor has wrong shape"
        assert y.shape[-1] == 1, "Target tensor should be a single value"

def test_random_forest_prep_split_and_target():
    # generate simple data
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=40, freq="H"),
        "reading_value_hourly_mean": range(40)
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        df.to_csv(os.path.join(tmpdir, "sensor.csv"), index=False)

        prep = RandomForestPrep(
            datapath=tmpdir,
            timestamp_col="timestamp",
            value_col="reading_value_hourly_mean",
            target_col="sensor",
            split_date="2023-01-01 20:00:00",
            horizon=1
        )

        train, test = prep.getdfs()
        assert "target" in train.columns, "Target column missing in training set"
        assert not train.isna().any().any(), "Training set contains NaNs"
