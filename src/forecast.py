import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

def load_and_forecast_SARIMA(
    model_path: str,
    test: pd.Series,
    exog_test: pd.DataFrame = None,
    plot: bool = False,
    y_limit_pct: float = 1.0,
    plot_path: str = None
) -> dict:
    """
    Load a saved SARIMAX model, forecast for the test period, optionally plot results,
    and return evaluation metrics.

    Args:
        model_path (str): Path to the saved SARIMAXResults pickle.
        test (pd.Series): Test series indexed by datetime.
        exog_test (pd.DataFrame, optional): Exogenous regressors for the test period.
        plot (bool): If True, plots actual vs forecast with confidence intervals.
        y_limit_pct (float): Proportion (0–1) of the combined series range to display on y-axis.
        plot_path (str, optional): File path to save the plot (e.g. "outputs/forecast.png").
    
    Returns:
        dict: Forecast metrics {"rmse": float, "mae": float}
    """
    # Load the model
    results = SARIMAXResults.load(model_path)
    
    # Forecast
    n_steps = len(test)
    fc = results.get_forecast(steps=n_steps, exog=exog_test)
    mean_pred = fc.predicted_mean
    conf_int = fc.conf_int()

    # Compute evaluation metrics
    rmse = root_mean_squared_error(test, mean_pred)
    mae = mean_absolute_error(test, mean_pred)
    metrics = {"rmse": rmse, "mae": mae}

    # Plot if requested
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(test.index, test, label='Actual')
        ax.plot(mean_pred.index, mean_pred, label='Forecast')
        ax.fill_between(
            conf_int.index,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            alpha=0.3,
            label='95% CI'
        )
        ax.legend()
        ax.set_title("SARIMA Forecast vs Actual")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")

        # Adjust y-axis range
        combined = pd.concat([test, mean_pred])
        y_min, y_max = combined.min(), combined.max()
        display_range = (y_max - y_min) * y_limit_pct
        ax.set_ylim(y_min, y_min + display_range)

        plt.tight_layout()

        # Save plot if specified
        if plot_path:
            save_dir = os.path.dirname(plot_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(plot_path)
            print(f"Plot saved to: {plot_path}")
        else:
            plt.show()
    
    return metrics