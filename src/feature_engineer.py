import pandas as pd
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

############### SARIMA ###############

class SARIMAprep:
    def __init__(
        self,
        data_path: str,
        timestamp_col: str,
        split_timestamp: str,
        value_col: str,
        method: str = "rolling_mean",
        window: str = "24H"
    ):
        """
        Args:
            data_path: Path to CSV file.
            timestamp_col: Name of the datetime column in the CSV.
            split_timestamp: ISO string (e.g. "2023-12-31") at which to split train/test.
            value_col: Column to impute.
            method: "ffill" or "rolling_mean".
            window: Pandas offset alias for rolling window (if method="rolling_mean").
        """
        self.data_path = data_path
        self.timestamp_col = timestamp_col
        self.split_timestamp = pd.to_datetime(split_timestamp)
        self.value_col = value_col
        self.method = method
        self.window = window

        # placeholders
        self.df: pd.DataFrame = None
        self.train: pd.DataFrame = None
        self.test: pd.DataFrame = None

        self._load_data()
        self._preprocess_index()
        self._split_data()
        self._impute_splits()

    def _load_data(self):
        self.df = pd.read_csv(self.data_path)

    def _preprocess_index(self):
        self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col])
        self.df.set_index(self.timestamp_col, inplace=True)
        self.df.sort_index(inplace=True)

    def _split_data(self):
        idx = self.df.index
        ts = self.split_timestamp

        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            if ts.tzinfo is None:
                ts = ts.tz_localize(idx.tz)
            else:
                ts = ts.tz_convert(idx.tz)
        elif isinstance(idx, pd.DatetimeIndex) and idx.tz is None and ts.tzinfo is not None:
            ts = ts.tz_convert(None)

        self.train = self.df.loc[:ts].copy()
        self.test  = self.df.loc[ts:].copy()

    @staticmethod
    def causal_impute(
        df: pd.DataFrame,
        value_col: str,
        method: str = "rolling_mean",
        window: str = "24H"
    ) -> pd.DataFrame:
        df_imputed = df.copy()
        if not isinstance(df_imputed.index, pd.DatetimeIndex):
            df_imputed.index = pd.to_datetime(df_imputed.index)

        series = df_imputed[value_col]

        if method == "ffill":
            # use .ffill() to avoid the future deprecation warning
            df_imputed[value_col] = series.ffill()

        elif method == "rolling_mean":
            past_mean = series.rolling(window=window, min_periods=1).mean()
            df_imputed[value_col] = series.fillna(past_mean)

        else:
            raise ValueError(f"Unknown method '{method}'; use 'ffill' or 'rolling_mean'.")

        return df_imputed

    def _impute_splits(self):
        self.train = self.causal_impute(self.train, self.value_col, self.method, self.window)
        self.train = self.train.dropna(subset=[self.value_col])
        self.test  = self.causal_impute(self.test,  self.value_col, self.method, self.window)
        self.test  = self.test.dropna(subset=[self.value_col])

    def get_train_test(self):
        return self.train, self.test

############### TLSTM ###############

class TLSTMprep:
    def __init__(
        self,
        datapath: str,
        lookback: int,
        horizon: int,
        batch_size: int,
        timestamp_col: str = "timestamp",
        value_col: str = "reading_value_hourly_mean",
        target_col: str = None,
        split_date: str = None,
        test_size: float = 0.2,
        shuffle_train: bool = True,
        hurdle: bool = False,
        delta: bool = False
    ):
        self.lookback   = lookback
        self.horizon    = horizon
        self.batch_size = batch_size
        self.target_col = target_col
        self.hurdle     = hurdle
        self.delta      = delta

        # 1. load & merge all CSVs
        df = self.load_and_merge_dfs(datapath, timestamp_col, value_col)

        # 2. drop any rows with NaN
        df = df.dropna()

        # 3. elapsed_time between rows (in hours)
        df["elapsed_time"] = (
            df.index.to_series()
              .diff()
              .dt.total_seconds()
              .div(3600)
              .fillna(0)
        )

        # 4. create target by shifting
        tgt = self.target_col or df.columns[0]
        df["target"] = df[tgt].shift(-self.horizon)

        # 4b. optionally create occurrence indicator
        if self.hurdle:
            df["occurrence"] = (df["target"] > 0).astype(int)

        # 4c. if delta=True, convert each feature into its diff and drop the original
        if self.delta:
            # identify feature columns (exclude elapsed_time, target, occurrence)
            feats = [
                col for col in df.columns
                if col not in ("elapsed_time", "target", "occurrence")
            ]
            # for each, make a delta column then drop the raw
            for f in feats:
                df[f + "_delta"] = df[f].diff()
            # df.drop(columns=feats, inplace=True)

        # 3b. one-hot encode month and hour
        df["month"] = df.index.month
        df["hour"]  = df.index.hour
        month_dummies = pd.get_dummies(df["month"], prefix="m", dtype=int)
        hour_dummies  = pd.get_dummies(df["hour"],  prefix="h", dtype=int)
        df = pd.concat([df, month_dummies, hour_dummies], axis=1)
        df.drop(columns=["month", "hour"], inplace=True)

        # drop rows with any NaN
        df = df.dropna()
        self.df = df

        # 5. train/test split
        if split_date is not None:
            split_ts = pd.to_datetime(split_date)
            idx = df.index
            if idx.tz is not None and split_ts.tzinfo is None:
                split_ts = split_ts.tz_localize(idx.tz)
            elif idx.tz is None and split_ts.tzinfo is not None:
                split_ts = split_ts.tz_convert(None)

            train_df = df.loc[:split_ts]
            test_df  = df.loc[split_ts:]
        else:
            n_test   = int(len(df) * test_size)
            train_df = df.iloc[:-n_test]
            test_df  = df.iloc[-n_test:]

        # 6. build DataLoaders
        self.train_loader = self._make_loader(train_df, shuffle=shuffle_train)
        self.test_loader  = self._make_loader(test_df,  shuffle=False)

    @staticmethod
    def load_and_merge_dfs(datapath: str, timestamp_col: str, value_col: str) -> pd.DataFrame:
        csv_files = glob.glob(os.path.join(datapath, "*.csv"))
        dfs = []
        for f in csv_files:
            name = os.path.splitext(os.path.basename(f))[0]
            tmp = pd.read_csv(f)
            tmp[timestamp_col] = pd.to_datetime(tmp[timestamp_col])
            tmp = tmp.set_index(timestamp_col)
            tmp = tmp[[value_col]].rename(columns={value_col: name})
            dfs.append(tmp)
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, axis=1, join="outer").sort_index()

    def _make_loader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        """
        Build overlapping sequences of length `lookback`:
        - X : all features except elapsed_time, target, occurrence
        - T : elapsed_time
        - y_amt : numeric target
        - y_occ : binary occurrence (if hurdle=True)
        """
        cols = list(df.columns)
        idx_elapsed = cols.index("elapsed_time")
        idx_target  = cols.index("target")
        idx_occ     = cols.index("occurrence") if "occurrence" in cols else None

        # always exclude only these three; if delta=True then "delta" is in cols and
        # so ends up in feature_cols, otherwise it's simply not in cols.
        exclude = {"elapsed_time", "target", "occurrence"}
        feature_cols = [c for c in cols if c not in exclude]
        feature_idxs = [cols.index(c) for c in feature_cols]

        arr = df.values
        seq_len = self.lookback

        Xs, Ts, y_amt, y_occ = [], [], [], []
        for start in range(len(arr) - seq_len + 1):
            window = arr[start : start + seq_len]
            Xs.append(window[:, feature_idxs])
            Ts.append(window[:, idx_elapsed : idx_elapsed + 1])
            y_amt.append(window[-1, idx_target])
            if self.hurdle:
                y_occ.append(window[-1, idx_occ])

        class SeqDataset(Dataset):
            def __init__(self, Xs, Ts, amt, occ, hurdle):
                self.Xs, self.Ts, self.amt, self.occ = Xs, Ts, amt, occ
                self.hurdle = hurdle

            def __len__(self):
                return len(self.amt)

            def __getitem__(self, idx):
                x = torch.tensor(self.Xs[idx], dtype=torch.float32)
                t = torch.tensor(self.Ts[idx], dtype=torch.float32)
                y_amount = torch.tensor([self.amt[idx]], dtype=torch.float32)
                if self.hurdle:
                    y_occurrence = torch.tensor([self.occ[idx]], dtype=torch.float32)
                    return x, t, y_amount, y_occurrence
                else:
                    return x, t, y_amount

        ds = SeqDataset(Xs, Ts, y_amt, y_occ, self.hurdle)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def __repr__(self):
        return (
            f"TLSTMprep(lookback={self.lookback}, horizon={self.horizon}, "
            f"batch_size={self.batch_size}, total_rows={len(self.df)}, "
            f"hurdle={self.hurdle}, delta={self.delta})"
        )

class TLSTMOriginalPrep:
    def __init__(
        self,
        datapath: str,
        lookback: int,
        horizon: int,
        batch_size: int,
        timestamp_col: str = "timestamp",
        value_col: str = "reading_value_hourly_mean",
        target_col: str = None,
        split_date: str = None,
        test_size: float = 0.2,
        shuffle_train: bool = True,
        delta: bool = False
    ):
        self.lookback   = lookback
        self.horizon    = horizon
        self.batch_size = batch_size
        self.target_col = target_col
        self.delta      = delta

        # 1. load & merge all CSVs
        df = self.load_and_merge_dfs(datapath, timestamp_col, value_col)

        # 2. drop any rows with NaN
        df = df.dropna()

        # 3. elapsed_time between rows (in hours)
        df["elapsed_time"] = (
            df.index.to_series()
              .diff()
              .dt.total_seconds()
              .div(3600)
              .fillna(0)
        )

        # 4. create target by shifting
        tgt = self.target_col or df.columns[0]
        df["target"] = df[tgt].shift(-self.horizon)

        # 4c. if delta=True, convert each feature into its diff and drop the original
        if self.delta:
            # identify feature columns (exclude elapsed_time, target, occurrence)
            feats = [
                col for col in df.columns
                if col not in ("elapsed_time", "target", "occurrence")
            ]
            # for each, make a delta column then drop the raw
            for f in feats:
                df[f + "_delta"] = df[f].diff()
            # df.drop(columns=feats, inplace=True)

        # 3b. one-hot encode month and hour
        df["month"] = df.index.month
        df["hour"]  = df.index.hour
        month_dummies = pd.get_dummies(df["month"], prefix="m", dtype=int)
        hour_dummies  = pd.get_dummies(df["hour"],  prefix="h", dtype=int)
        df = pd.concat([df, month_dummies, hour_dummies], axis=1)
        df.drop(columns=["month", "hour"], inplace=True)

        # drop rows with any NaN (handles shift/diff)
        df = df.dropna()
        self.df = df

        # 5. train/test split
        if split_date is not None:
            split_ts = pd.to_datetime(split_date)
            idx = df.index
            if idx.tz is not None and split_ts.tzinfo is None:
                split_ts = split_ts.tz_localize(idx.tz)
            elif idx.tz is None and split_ts.tzinfo is not None:
                split_ts = split_ts.tz_convert(None)
            train_df = df.loc[:split_ts]
            test_df  = df.loc[split_ts:]
        else:
            n_test   = int(len(df) * test_size)
            train_df = df.iloc[:-n_test]
            test_df  = df.iloc[-n_test:]

        # 6. build DataLoaders
        self.train_loader = self._make_loader(train_df, shuffle=shuffle_train)
        self.test_loader  = self._make_loader(test_df,  shuffle=False)

    @staticmethod
    def load_and_merge_dfs(datapath: str, timestamp_col: str, value_col: str) -> pd.DataFrame:
        csv_files = glob.glob(os.path.join(datapath, "*.csv"))
        dfs = []
        for f in csv_files:
            name = os.path.splitext(os.path.basename(f))[0]
            tmp = pd.read_csv(f)
            tmp[timestamp_col] = pd.to_datetime(tmp[timestamp_col])
            tmp = tmp.set_index(timestamp_col)
            tmp = tmp[[value_col]].rename(columns={value_col: name})
            dfs.append(tmp)
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, axis=1, join="outer").sort_index()

    def _make_loader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        """
        Build overlapping sequences of length `lookback`:
        - X : all features except elapsed_time and target
        - T : elapsed_time
        - y  : numeric target
        """
        cols = list(df.columns)
        idx_elapsed = cols.index("elapsed_time")
        idx_target  = cols.index("target")

        # exclude only elapsed_time and target; delta stays if created
        exclude = {"elapsed_time", "target"}
        feature_cols = [c for c in cols if c not in exclude]
        feature_idxs = [cols.index(c) for c in feature_cols]

        arr = df.values
        L   = self.lookback

        Xs, Ts, Ys = [], [], []
        for i in range(len(arr) - L + 1):
            window = arr[i : i + L]
            Xs.append(window[:, feature_idxs])
            Ts.append(window[:, idx_elapsed : idx_elapsed + 1])
            Ys.append(window[-1, idx_target])

        class SeqDataset(Dataset):
            def __init__(self, Xs, Ts, Ys):
                self.Xs, self.Ts, self.Ys = Xs, Ts, Ys

            def __len__(self):
                return len(self.Ys)

            def __getitem__(self, idx):
                x = torch.tensor(self.Xs[idx], dtype=torch.float32)
                t = torch.tensor(self.Ts[idx], dtype=torch.float32)
                y = torch.tensor([self.Ys[idx]], dtype=torch.float32)
                return x, t, y

        ds = SeqDataset(Xs, Ts, Ys)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def __repr__(self):
        return (
            f"TLSTMOriginalPrep(lookback={self.lookback}, horizon={self.horizon}, "
            f"batch_size={self.batch_size}, total_rows={len(self.df)}, "
            f"delta={self.delta})"
        )
    
############### TimeXer ###############

class TimeXerDataLoader:
    def __init__(
        self,
        datapath: str,
        timestamp_col: str,
        value_col: str,
        feature_cols: list[str],
        target_col: str,
        window_size: int = 24,
        prediction_length: int = 1,
        split_date: pd.Timestamp = pd.Timestamp("2024-01-01", tz="Asia/Singapore"),
        batch_size: int = 64,
        num_workers: int = 2,
    ):
        self.datapath = datapath
        self.timestamp_col = timestamp_col
        self.value_col = value_col
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.window_size = window_size
        self.prediction_length = prediction_length
        self.split_date = split_date
        self.batch_size = batch_size
        self.num_workers = num_workers

    def load_and_merge_dfs(self) -> pd.DataFrame:
        csv_files = glob.glob(os.path.join(self.datapath, "*.csv"))
        dfs = []
        for f in csv_files:
            name = os.path.splitext(os.path.basename(f))[0]
            tmp = pd.read_csv(f)
            tmp[self.timestamp_col] = pd.to_datetime(tmp[self.timestamp_col])
            tmp = tmp.set_index(self.timestamp_col)
            tmp = tmp[[self.value_col]].rename(columns={self.value_col: name})
            dfs.append(tmp)
        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs, axis=1, join="outer").sort_index()
        return df.dropna()

    def prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # elapsed time in hours
        df = df.copy()
        df["elapsed_time"] = (
            df.index.to_series()
               .diff()
               .dt.total_seconds()
               .div(3600)
               .fillna(0)
        )
        # integer time index
        df["time_idx"] = ((df.index - df.index.min()) 
                           / pd.Timedelta(hours=1)).astype(int)
        df["series_id"] = 0
        return df

    def make_dataset(self, df: pd.DataFrame) -> TimeSeriesDataSet:
        return TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=self.target_col,
            group_ids=["series_id"],
            min_encoder_length=self.window_size,
            max_encoder_length=self.window_size,
            min_prediction_length=self.prediction_length,
            max_prediction_length=self.prediction_length,
            time_varying_known_reals=["time_idx"] + self.feature_cols,
            time_varying_unknown_reals=[self.target_col],
            target_normalizer=GroupNormalizer(groups=["series_id"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

    def get_dataloaders(self):
        # load and preprocess
        df = self.load_and_merge_dfs()
        df = self.prepare_df(df)

        # split
        train_df = df[df.index <= self.split_date]
        test_df  = df[df.index  > self.split_date]

        # build datasets
        training  = self.make_dataset(train_df)
        validation = self.make_dataset(test_df)

        # build loaders
        train_loader = training.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        val_loader = validation.to_dataloader(
            train=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        return train_loader, val_loader
    
class RandomForestPrep:
    def __init__(
        self,
        datapath: str,
        horizon: int,
        timestamp_col: str = "timestamp",
        value_col: str = "reading_value_hourly_mean",
        target_col: str = None,
        split_date: str = None,
        test_size: float = 0.2,
        delta: bool = False
    ):
        self.horizon    = horizon
        self.target_col = target_col
        self.delta      = delta

        # 1. load & merge all CSVs
        df = self.load_and_merge_dfs(datapath, timestamp_col, value_col)

        # 2. drop any rows with NaN
        df = df.dropna()

        # 3. elapsed_time between rows (in hours)
        df["elapsed_time"] = (
            df.index.to_series()
              .diff()
              .dt.total_seconds()
              .div(3600)
              .fillna(0)
        )

        # 4. create target by shifting
        tgt = self.target_col or df.columns[0]
        df["target"] = df[tgt].shift(-self.horizon)

        # 4c. if delta=True, convert each feature into its diff and drop the original
        if self.delta:
            # identify feature columns (exclude elapsed_time, target, occurrence)
            feats = [
                col for col in df.columns
                if col not in ("elapsed_time", "target", "occurrence")
            ]
            # for each, make a delta column then drop the raw
            for f in feats:
                df[f + "_delta"] = df[f].diff()
            # df.drop(columns=feats, inplace=True)

        # 3b. one-hot encode month and hour
        df["month"] = df.index.month
        df["hour"]  = df.index.hour
        month_dummies = pd.get_dummies(df["month"], prefix="m", dtype=int)
        hour_dummies  = pd.get_dummies(df["hour"],  prefix="h", dtype=int)
        df = pd.concat([df, month_dummies, hour_dummies], axis=1)
        df.drop(columns=["month", "hour"], inplace=True)

        # drop rows with any NaN (handles shift/diff)
        df = df.dropna()
        self.df = df

        # 5. train/test split
        if split_date is not None:
            split_ts = pd.to_datetime(split_date)
            idx = df.index
            if idx.tz is not None and split_ts.tzinfo is None:
                split_ts = split_ts.tz_localize(idx.tz)
            elif idx.tz is None and split_ts.tzinfo is not None:
                split_ts = split_ts.tz_convert(None)
            self.train_df = df.loc[:split_ts]
            self.test_df  = df.loc[split_ts:]
        else:
            n_test   = int(len(df) * test_size)
            self.train_df = df.iloc[:-n_test]
            self.test_df  = df.iloc[-n_test:]

    @staticmethod
    def load_and_merge_dfs(datapath: str, timestamp_col: str, value_col: str) -> pd.DataFrame:
        csv_files = glob.glob(os.path.join(datapath, "*.csv"))
        dfs = []
        for f in csv_files:
            name = os.path.splitext(os.path.basename(f))[0]
            tmp = pd.read_csv(f)
            tmp[timestamp_col] = pd.to_datetime(tmp[timestamp_col])
            tmp = tmp.set_index(timestamp_col)
            tmp = tmp[[value_col]].rename(columns={value_col: name})
            dfs.append(tmp)
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, axis=1, join="outer").sort_index()

    def getdfs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return:
        train and test df
        """
        return self.train_df, self.test_df