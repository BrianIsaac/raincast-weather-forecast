# Standard library imports
from datetime import datetime
import glob
import os
from typing import Optional, Tuple

# Third-party imports
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import plotly.graph_objects as go
import requests
from pyampute.exploration.mcar_statistical_tests import MCARTest
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, ccf, kpss

############### Functions ###############

def download_datasets_by_type(collection_id, data_type):
    # Get metadata
    meta_url = f"https://api-production.data.gov.sg/v2/public/api/collections/{collection_id}/metadata"
    response = requests.get(meta_url)
    response.raise_for_status()
    data = response.json()
    
    # Extract collection metadata
    metadata = data.get('data', {}).get('collectionMetadata', {})
    coverage_start = metadata.get('coverageStart')
    coverage_end = metadata.get('coverageEnd')
    child_datasets = metadata.get('childDatasets', [])
    
    # Convert start and end to years
    start_year = datetime.fromisoformat(coverage_start).year if coverage_start else None
    end_year = datetime.fromisoformat(coverage_end).year if coverage_end else None
    
    # Create year:dataset_id mapping
    year_dataset_dict = {}
    if start_year and end_year and len(child_datasets) == (end_year - start_year + 1):
        for i, dataset_id in enumerate(child_datasets):
            year = start_year + i
            year_dataset_dict[year] = dataset_id
    else:
        print("Warning: Number of datasets does not match year range.")
        return

    # Create directory
    save_dir = f"./data/{data_type}"
    os.makedirs(save_dir, exist_ok=True)

    # Base download initiation URL
    base_url = "https://api-open.data.gov.sg/v1/public/api/datasets/{}/initiate-download"

    # Download each dataset
    for year, dataset_id in year_dataset_dict.items():
        url = base_url.format(dataset_id)
        try:
            # Use GET to initiate download
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            download_url = data.get("data", {}).get("url")
            
            if download_url:
                # Download the file
                file_response = requests.get(download_url)
                file_response.raise_for_status()
                
                # Define file path
                filename = f"Historical{data_type}acrossSingapore{year}.csv"
                file_path = os.path.join(save_dir, filename)
                
                # Write content
                with open(file_path, "wb") as f:
                    f.write(file_response.content)
                
                print(f"Downloaded {filename} successfully.")
            else:
                print(f"Download URL not found for {year}.")
        
        except Exception as e:
            print(f"Failed to download data for {year}: {e}")

def missingness_report(df: pd.DataFrame, figsize=(12, 6)) -> pd.DataFrame:
    """
    1) Summarize missingness per column.
    2) Visualize missingness:
       - If there's exactly one numeric column, plot interactively with Plotly.
       - If there are multiple numeric columns, plot each one separately with Plotly.
       - Otherwise (no numeric cols), show missingno matrix and bar chart.
    3) Perform Little's MCAR test using pyampute if there are at least two numeric columns.

    Returns:
        pd.DataFrame: summary table with columns ['missing_count','missing_pct'].
    """
    # --- 1) Missingness summary on all columns ---
    miss_count = df.isnull().sum()
    miss_pct   = df.isnull().mean() * 100
    summary = (
        pd.DataFrame({
            "missing_count": miss_count,
            "missing_pct":   miss_pct.round(2)
        })
        .sort_values("missing_pct", ascending=False)
    )
    print("Missingness Summary:")
    print(summary)

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    # --- 2) Visualization ---
    if len(numeric_cols) == 1:
        # Univariate series
        col = numeric_cols[0]
        series = df[col].sort_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, mode='lines', name=col, line=dict(width=1)
        ))
        missing = series[series.isnull()]
        if not missing.empty:
            y0 = series.min() if pd.notnull(series.min()) else 0
            fig.add_trace(go.Scatter(
                x=missing.index, y=[y0]*len(missing),
                mode='markers', name='Missing',
                marker=dict(symbol='x', color='red', size=6)
            ))
        fig.update_layout(
            title=f"{col} (hourly) with Missing Points",
            xaxis_title="Timestamp", yaxis_title=col,
            hovermode="x unified"
        )
        fig.show()

    elif len(numeric_cols) > 1:
        # Multivariate: one chart per feature
        for col in numeric_cols:
            series = df[col].sort_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values, mode='lines', name=col, line=dict(width=1)
            ))
            missing = series[series.isnull()]
            if not missing.empty:
                y0 = series.min() if pd.notnull(series.min()) else 0
                fig.add_trace(go.Scatter(
                    x=missing.index, y=[y0]*len(missing),
                    mode='markers', name='Missing',
                    marker=dict(symbol='x', color='red', size=6)
                ))
            fig.update_layout(
                title=f"{col} (hourly) with Missing Points",
                xaxis_title="Timestamp", yaxis_title=col,
                hovermode="x unified"
            )
            fig.show()

    else:
        # No numeric columns: missingno fallback
        plt.figure(figsize=figsize)
        msno.matrix(df)
        plt.title("Missingness Matrix")
        plt.show()

        plt.figure(figsize=figsize)
        msno.bar(df)
        plt.title("Missingness Bar Chart")
        plt.show()

    # --- 3) Little's MCAR test via pyampute ---
    if len(numeric_cols) < 2:
        print("MCAR test requires at least two numeric columns; skipping.")
    else:
        try:
            mt = MCARTest(method="little")
            # Pass the DataFrame, not a numpy array:
            p_val = mt.little_mcar_test(df[numeric_cols])
            print(f"Little's MCAR test p-value = {p_val:.3f}")
            if p_val > 0.05:
                print("Little's test: data plausibly MCAR (fail to reject H0).")
            else:
                print("Little's test: data may not be MCAR (reject H0).")
        except ImportError:
            print("pyampute not installed; skipping MCAR test.")
        except Exception as e:
            print(f"MCAR test error: {e}")

    return summary

def causal_impute(
    df: pd.DataFrame,
    value_col: str,
    method: str = "rolling_mean",
    window: str = "24H"
) -> pd.DataFrame:
    """
    Impute missing values using only past data (no peeking ahead), minimizing leakage.

    Args:
        df (pd.DataFrame): Input DataFrame indexed by a DatetimeIndex.
        value_col (str): Name of the column to impute.
        method (str): One of:
            - "ffill": forward-fill last observed value.
            - "rolling_mean": fill missing with the mean over the past `window`.
        window (str): Rolling window size (pandas offset alias), used if method="rolling_mean".

    Returns:
        pd.DataFrame: A copy of `df` with `value_col` imputed.
    """
    # Work on a copy, ensure DatetimeIndex
    df_imputed = df.copy()
    if not isinstance(df_imputed.index, pd.DatetimeIndex):
        df_imputed.index = pd.to_datetime(df_imputed.index)

    series = df_imputed[value_col]

    if method == "ffill":
        # Forward-fill uses only past values
        df_imputed[value_col] = series.fillna(method="ffill")

    elif method == "rolling_mean":
        # Compute past rolling mean (causal)
        # min_periods=1 means the first non-NA will seed the average
        past_mean = series.rolling(window=window, min_periods=1).mean()
        # Fill missing points with the past rolling mean
        df_imputed[value_col] = series.fillna(past_mean)

    else:
        raise ValueError(f"Unknown method '{method}'; use 'ffill' or 'rolling_mean'.")

    return df_imputed

def load_and_merge_dfs(datapath: str) -> pd.DataFrame:
    """
    Load all CSV files in the given datapath, set 'timestamp' as the datetime index,
    rename 'reading_value_hourly_mean' to the CSV filename (without extension),
    and perform a full outer join on the timestamp index across all DataFrames.

    Parameters:
    - datapath (str): Path to the directory containing CSV files.

    Returns:
    - pd.DataFrame: Merged DataFrame with 'timestamp' as a column and each CSV's readings as separate columns.
    """
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(datapath, '*.csv'))
    dfs = []

    for file in csv_files:
        # Derive column name from the filename
        name = os.path.splitext(os.path.basename(file))[0]

        # Load CSV
        df = pd.read_csv(file)

        # Convert and set datetime index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Rename the reading column
        df = df[['reading_value_hourly_mean']].rename(
            columns={'reading_value_hourly_mean': name}
        )

        dfs.append(df)

    # If no DataFrames found, return empty DataFrame
    if not dfs:
        return pd.DataFrame()

    # Perform full outer join on index
    final_df = pd.concat(dfs, axis=1, join='outer')

    # Reset index to turn 'timestamp' back into a column
    final_df = final_df.reset_index()
    final_df.set_index('timestamp', inplace=True)

    return final_df

def clean_multivariate(
    df: pd.DataFrame,
    drop_year: int = 2016
) -> Tuple[pd.DataFrame, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]:
    """
    1) Remove all data from `drop_year`.
    2) Forward-fill every column (no future leakage).
    3) Drop any remaining NaN rows (the initial period that ffill couldn't fill).
    4) Return the cleaned df plus the (start, end) timestamps of that dropped period.

    Parameters
    ----------
    df : pd.DataFrame
        Datetime‐indexed DataFrame to clean.
    drop_year : int, default=2016
        Year to completely remove before imputation.

    Returns
    -------
    cleaned_df : pd.DataFrame
        The DataFrame after dropping, ffill, and dropping residual NaNs.
    dropped_window : (start, end)
        A tuple of pd.Timestamp giving the first and last timestamps that were
        removed because they contained initial NaNs. If nothing was dropped, both
        will be None.
    """
    # 1) Drop the specified year
    df2 = df[df.index.year != drop_year].copy()

    # 2) Forward-fill all columns
    df_ffill = df2.ffill()

    # 3) Identify which rows still have any NaNs
    mask_valid   = df_ffill.notna().all(axis=1)
    dropped_idx  = df_ffill.index[~mask_valid]

    if not dropped_idx.empty:
        dropped_start = dropped_idx.min()
        dropped_end   = dropped_idx.max()
    else:
        dropped_start = dropped_end = None

    # 4) Final cleaned DataFrame
    cleaned_df = df_ffill.loc[mask_valid]

    return cleaned_df, (dropped_start, dropped_end)


############### Classes ###############

class DataPreparation:
    """
    A class for preparing time-series data by aggregating readings,
    either from a single DataFrame or by loading all CSVs in a folder.
    """

    def __init__(
        self, 
        df: pd.DataFrame = None, 
        folder_path: str = None, 
        timestamp_col: str = "timestamp", 
        value_col: str = "reading_value"
    ):
        """
        Initializes the DataPreparation object.
        Either `df` or `folder_path` must be provided.

        Args:
            df (pd.DataFrame): DataFrame with raw readings.
            folder_path (str): Path to folder containing CSV files to load & concatenate.
            timestamp_col (str): Column name for timestamps.
            value_col (str): Column name for reading values.
        """
        if folder_path:
            # load and concatenate all CSVs in the folder
            files = glob.glob(os.path.join(folder_path, "*.csv"))
            if not files:
                raise FileNotFoundError(f"No CSV files found in {folder_path}")
            df_list = [pd.read_csv(fp) for fp in files]
            self.df = pd.concat(df_list, ignore_index=True)
        elif df is not None:
            self.df = df.copy()
        else:
            raise ValueError("Must provide either a DataFrame or a folder_path")

        self.timestamp_col = timestamp_col
        self.value_col = value_col

    def _ensure_datetime_index(self):
        """Convert the timestamp column to pandas datetime index."""
        self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col])
        self.df.set_index(self.timestamp_col, inplace=True)
        return self

    def aggregate_mean_by_timestamp(self) -> pd.DataFrame:
        """
        Aggregates the reading values by exact timestamp, computing the mean for each timestamp.

        Returns:
            pd.DataFrame: Indexed by timestamp with column '<value_col>_mean'.
        """
        self._ensure_datetime_index()
        ts_mean = self.df.groupby(level=0)[self.value_col].mean()
        ts_mean.name = f"{self.value_col}_mean"
        return ts_mean.to_frame()

    def aggregate_mean_by_hour(self) -> pd.DataFrame:
        """
        First aggregates by exact timestamp (mean), then resamples by hour to compute hourly means.

        Returns:
            pd.DataFrame: Indexed by hour with column '<value_col>_hourly_mean'.
        """
        ts_mean = self.aggregate_mean_by_timestamp()
        hourly = ts_mean.resample('h').mean()
        hourly.rename(columns={ts_mean.columns[0]: f"{self.value_col}_hourly_mean"}, inplace=True)
        return hourly
    
class UnivariateEDA:
    """
    Perform univariate EDA on a single time series with a DatetimeIndex.

    Parameters:
    -----------
    df : pd.DataFrame or pd.Series
        If DataFrame, must contain exactly one column (or specify `value_col`).
        Must be indexed by a pd.DatetimeIndex.
    value_col : str, optional
        Column name to extract if df is a DataFrame with multiple columns.
    seasonal_period : int, optional
        Seasonal period (s) for seasonal differencing and seasonal ACF/PACF.
    """
    def __init__(self, df, value_col=None, seasonal_period=None):
        if isinstance(df, pd.Series):
            series = df.copy()
        else:
            if value_col:
                series = df[value_col].copy()
            elif df.shape[1] == 1:
                series = df.iloc[:,0].copy()
            else:
                raise ValueError("DataFrame must have one column or specify value_col.")
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Index must be a pd.DatetimeIndex.")
        self.series = series.asfreq(series.index.inferred_freq or 'H')
        self.seasonal_period = seasonal_period

    def check_stationarity(self, alpha=0.05):
        """
        Runs ADF and KPSS tests and prints results.
        Returns a dict with test statistics and p-values.
        """
        x = self.series.dropna()
        # ADF test (null: non-stationary)
        adf_res = adfuller(x, autolag='AIC')
        # KPSS test (null: stationary)
        kpss_res = kpss(x, nlags="auto", regression='c')
        print(">>> Augmented Dickey-Fuller Test")
        print(f"ADF Statistic: {adf_res[0]:.4f}")
        print(f"p-value: {adf_res[1]:.4f}")
        print("Critical Values:")
        for k, v in adf_res[4].items():
            print(f"  {k}: {v:.4f}")
        print("\n>>> KPSS Test")
        print(f"KPSS Statistic: {kpss_res[0]:.4f}")
        print(f"p-value: {kpss_res[1]:.4f}")
        print("Critical Values:")
        for k, v in kpss_res[3].items():
            print(f"  {k}: {v:.4f}")
        return {
            "adf_stat": adf_res[0], "adf_p": adf_res[1],
            "kpss_stat": kpss_res[0], "kpss_p": kpss_res[1]
        }

    def difference(self, d=0, D=0):
        """
        Apply non-seasonal differencing (d) and seasonal differencing (D).
        Returns a new pd.Series.
        """
        s = self.series.copy()
        # non-seasonal differencing
        for _ in range(d):
            s = s.diff()
        # seasonal differencing
        if D and self.seasonal_period:
            for _ in range(D):
                s = s.diff(self.seasonal_period)
        return s.dropna()

    def plot_acf_pacf(self, lags=40, seasonal_lags=None):
        """
        Plot ACF and PACF for the series.
        If seasonal_period is set, can pass seasonal_lags (e.g. s*2, s*3).
        """
        s = self.series.dropna()
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(s, lags=lags, ax=axes[0], title="ACF")
        plot_pacf(s, lags=lags, ax=axes[1], title="PACF")
        plt.tight_layout()
        plt.show()
        if self.seasonal_period and seasonal_lags:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            plot_acf(s, lags=seasonal_lags, ax=axes[0],
                     title=f"Seasonal ACF (lags up to {seasonal_lags})")
            plot_pacf(s, lags=seasonal_lags, ax=axes[1],
                      title=f"Seasonal PACF (lags up to {seasonal_lags})")
            plt.tight_layout()
            plt.show()

class MultivariateEDA:
    """
    Multivariate EDA for +1hr rainfall forecasting features.
    Assumes df is datetime-indexed and contains:
      - predictors: rainfall, humidity, wind_direction, air_temperature, wind_speed
      - target_col: column name for rainfall (will be shifted internally)
    """
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df.copy()
        # create a +1hr ahead target series
        self.df['target'] = self.df[target_col].shift(-1)
        # include everything except the shifted 'target' as features
        self.features = [c for c in self.df.columns if c != 'target']

    def correlation_heatmap(self):
        corr = self.df[self.features + ['target']].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Matrix (features vs +1hr rainfall)")
        plt.show()

    def pairplot(self):
        sns.pairplot(self.df[self.features + ['target']].dropna())
        plt.suptitle("Pairplot of Features and Target", y=1.02)
        plt.show()

    def cross_correlation(self, feature: str, max_lag: int = 24):
        xs = self.df[feature].fillna(0)
        ys = self.df['target'].fillna(0)
        ccf_vals = ccf(xs, ys)[:max_lag+1]
        lags = list(range(max_lag+1))
        plt.figure(figsize=(8,4))
        plt.stem(lags, ccf_vals, basefmt=" ")
        plt.title(f"CCF: {feature} vs +1hr rainfall")
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        plt.show()