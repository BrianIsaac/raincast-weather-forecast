"""Utilities or functions that are useful across all the different
modules in this package can be defined here."""

import logging
import logging.config
import os
import shutil
import time
from typing import Any, Optional

import mlflow
import yaml

logger = logging.getLogger(__name__)


def setup_logging(
    logging_config_path="./conf/logging.yaml", default_level=logging.INFO, log_dir=None
):
    """Set up configuration for logging utilities.

    Parameters
    ----------
    logging_config_path : str, optional
        Path to YAML file containing configuration for Python logger,
        by default "./conf/logging.yaml"
    default_level : logging object, optional, by default logging.INFO
        log_dir : str, optional
        Directory to store log files, by default None (uses directory in config)
    """

    try:
        with open(logging_config_path, "rt", encoding="utf-8") as file:
            log_config = yaml.safe_load(file.read())

        # Modify log file paths if log_dir is provided
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            for handler in log_config["handlers"].values():
                if "filename" in handler:
                    # Extract just the filename from the path
                    filename = os.path.basename(handler["filename"])
                    # Create new path with the provided directory
                    handler["filename"] = os.path.join(log_dir, filename)

        logging.config.dictConfig(log_config)

    except Exception as error:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.error(error)
        logger.error("Logging config file is not found. Basic config is being used.")


def mlflow_init(
    tracking_uri, exp_name, run_name, setup_mlflow=False, autolog=False, resume=False
):
    """Initialise MLflow connection.

    Parameters
    ----------
    tracking_uri : string
        Tracking URI used for MLFlow
    exp_name : string
        Experiment name used for MLFlow
    run_name : string
        Run name for the experiment used for MLFlow
    setup_mlflow : bool, optional
        Choice to set up MLflow connection, by default False
    autolog : bool, optional
        Choice to set up MLflow's autolog, by default False
    resume : bool, optional
        Choice to resume using the latest previous run with the same
        name, by default False

    Returns
    -------
    init_success : bool
        Boolean value indicative of success
        of intialising connection with MLflow server.

    mlflow_run : Union[None, `mlflow.entities.Run` object]
        On successful initialisation, the function returns an object
        containing the data and properties for the MLflow run.
        On failure, the function returns a null value.

    step_offset : int
        The last step number from the previous run if resuming, or 0 if starting a new run.
        This can be used to continue logging metrics with incrementing step numbers.
    """
    init_success = False
    mlflow_run = None
    step_offset = 0

    if setup_mlflow:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(exp_name)
            mlflow.enable_system_metrics_logging()

            if autolog:
                mlflow.autolog()

            if "MLFLOW_HP_TUNING_TAG" in os.environ:
                run_name += "-hp"

            base_run_name = run_name
            client = mlflow.tracking.MlflowClient()

            if resume:
                # Try to find the most recent run with the same prefix name
                experiment = client.get_experiment_by_name(exp_name)
                if experiment:
                    runs = client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string=f"tags.mlflow.runName LIKE '{base_run_name}-%'",
                        order_by=["attribute.start_time DESC"],
                        max_results=1,
                    )
                    if runs:
                        # Resume the most recent run
                        run_id = runs[0].info.run_id
                        mlflow.start_run(run_id=run_id)
                        logger.info(f"Resuming previous run: {runs[0].info.run_name}")

                        # Find the maximum step across all metrics
                        run_data = client.get_run(run_id).data
                        max_steps = []

                        # Get all metric keys - handle different MLflow API versions
                        try:
                            # Try accessing metrics as a dictionary
                            metric_keys = run_data.metrics.keys()
                        except AttributeError:
                            try:
                                # Try accessing metrics as a list of objects with key attribute
                                metric_keys = [m.key for m in run_data.metrics]
                            except (AttributeError, TypeError):
                                # Fallback: get metric keys from metric history
                                logger.warning(
                                    "Could not directly access metric keys, using alternative method"
                                )
                                metric_keys = []
                                # Try to get all metrics by listing them from the run
                                try:
                                    for metric in client.list_metrics(run_id):
                                        metric_keys.append(metric.key)
                                except Exception as e:
                                    logger.warning(f"Error listing metrics: {e}")

                        # Get maximum step for each metric
                        for metric_key in metric_keys:
                            if not metric_key.startswith("system/"):
                                try:
                                    metric_history = client.get_metric_history(
                                        run_id, metric_key
                                    )
                                    if metric_history:
                                        max_steps.append(
                                            max(m.step for m in metric_history)
                                        )
                                except Exception as e:
                                    logger.warning(
                                        f"Error getting history for metric {metric_key}: {e}"
                                    )

                        if max_steps:
                            step_offset = max(max_steps)
                            logger.info(f"Continuing from step {step_offset}")
                    else:
                        # No previous run found, create a new one
                        run_name = f"{base_run_name}-{int(time.time())}"
                        mlflow.start_run(run_name=run_name)
                        logger.info(
                            f"No previous run found. Starting new run: {run_name}"
                        )
                else:
                    # Experiment not found, create a new run
                    run_name = f"{base_run_name}-{int(time.time())}"
                    mlflow.start_run(run_name=run_name)
            else:
                # Start a new run with timestamp
                run_name = f"{base_run_name}-{int(time.time())}"
                mlflow.start_run(run_name=run_name)

            def set_tag(env_var, tag_name=""):
                if env_var in os.environ:
                    key = tag_name if tag_name != "" else env_var.lower()
                    mlflow.set_tag(key, os.environ.get(env_var))

            set_tag("MLFLOW_HP_TUNING_TAG", "hptuning_tag")
            set_tag("JOB_UUID")
            set_tag("JOB_NAME")

            mlflow_run = mlflow.active_run()
            init_success = True
            logger.info("MLflow initialisation has succeeded.")
            logger.info("UUID for MLflow run: %s", mlflow_run.info.run_id)
        except Exception as e:
            logger.error("MLflow initialisation has failed.")
            logger.error(e)

    return init_success, mlflow_run, step_offset


def mlflow_log(mlflow_init_status, log_function, **kwargs):
    """Custom function for utilising MLflow's logging functions.

    This function is only relevant when the function `mlflow_init`
    returns a "True" value, translating to a successful initialisation
    of a connection with an MLflow server.

    Parameters
    ----------
    mlflow_init_status : bool
        Boolean value indicative of success of intialising connection
        with MLflow server.
    log_function : str
        Name of MLflow logging function to be used.
        See https://www.mlflow.org/docs/latest/python_api/mlflow.html
    **kwargs
        Keyword arguments passed to `log_function`.
    """
    if mlflow_init_status:
        try:
            method = getattr(mlflow, log_function)
            method(
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key in method.__code__.co_varnames
                }
            )
        except Exception as error:
            logger.error(error)


def mlflow_pytorch_call(
    mlflow_init_status: bool, pytorch_function: str, **kwargs
) -> Optional[Any]:
    """
    Convenience wrapper around the ``mlflow.pytorch`` API.

    This helper is intended to be used **only** after an MLflow tracking
    server / experiment has been successfully initialised, i.e. when the
    function that establishes the MLflow connection returns ``True``.
    It dynamically resolves the required *PyTorch–specific* MLflow
    function (e.g. ``log_model``, ``save_model``, ``load_model``) and
    forwards the provided keyword arguments if – and only if – they
    are part of the function’s formal parameter list.

    Parameters
    ----------
    mlflow_init_status : bool
        Flag indicating whether a connection to an MLflow tracking
        server has been successfully established.
    pytorch_function : str
        Name of the ``mlflow.pytorch`` function to be invoked.
        Refer to the official documentation:
        https://www.mlflow.org/docs/latest/api_reference/python_api/mlflow.pytorch.html
    **kwargs
        Arbitrary keyword arguments to be forwarded to the chosen
        ``mlflow.pytorch`` function.

    Returns
    -------
    Any or None
        Whatever the invoked ``mlflow.pytorch`` function returns.
        If ``mlflow_init_status`` is ``False`` or an exception is raised,
        ``None`` is returned.

    Notes
    -----
    • Only keyword arguments that appear in the target function’s
      signature are forwarded.
    • Errors are logged but **not** re‑raised to avoid interrupting the
      calling workflow.

    Examples
    --------
    >>> mlflow_pytorch_call(
    ...     mlflow_init_status=mlflow_init(...),
    ...     pytorch_function="log_model",
    ...     pytorch_model=model,
    ...     artifact_path="models/sketch"
    ... )
    """
    if not mlflow_init_status:
        return None

    try:
        method = getattr(mlflow.pytorch, pytorch_function)
    except AttributeError as err:
        logger.error(
            "Function '%s' does not exist in mlflow.pytorch: %s",
            pytorch_function,
            err,
        )
        return None

    try:
        # Forward only those kwargs that are accepted by the target method
        valid_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in method.__code__.co_varnames
        }
        return method(**valid_kwargs)
    except Exception as err:  # noqa: BLE001
        logger.error(
            "mlflow.pytorch.%s failed with error: %s",
            pytorch_function,
            err,
        )
        return None
    
def clear_directory(dir_path: str) -> None:
    """
    Remove all files and sub-directories in dir_path,
    but keep dir_path itself.
    """
    if not os.path.isdir(dir_path):
        logging.warning("Directory %s does not exist; nothing to clear.", dir_path)
        return

    for entry in os.listdir(dir_path):
        entry_path = os.path.join(dir_path, entry)
        try:
            if os.path.isfile(entry_path) or os.path.islink(entry_path):
                os.unlink(entry_path)
            elif os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
        except Exception as e:
            logging.error("Failed to delete %s: %s", entry_path, e)