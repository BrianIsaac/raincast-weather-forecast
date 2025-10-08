"""
This script is for training a model on singapore weather dataset.
"""
# Standard library
import json
import logging
import math
import os

# Third-party libraries
import pandas as pd
import hydra
import mlflow
from omegaconf import OmegaConf
import torch
import joblib

# Local application imports
import general_utils
from feature_engineer import TLSTMprep, SARIMAprep, TimeXerDataLoader, RandomForestPrep
from model import TLSTMEnsembleTrainer, fit_and_save_sarimax, TimeXerTrainer, RandomForestTrainer
from forecast import load_and_forecast_SARIMA

# pylint: disable = no-value-for-parameter
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(args):
    """This is the main function for training the model.

    Parameters
    ----------
    args : omegaconf.DictConfig
        An omegaconf.DictConfig object containing arguments for the main function.
    """

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    general_utils.setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        ),
        log_dir=args.get("log_dir", None),
    )

    ## Initialize MLflow (Use the appropriate function from general_utils.py)
    # mlflow_init_status, mlflow_run = general_utils.<method>
    mlflow_init_status, mlflow_run, step_offset = general_utils.mlflow_init(
        tracking_uri=args.mlflow_tracking_uri, 
        exp_name=args.mlflow_exp_name, 
        run_name=args.mlflow_run_name, 
        setup_mlflow=True, 
        autolog=True,
        resume=args.resume
    )

#### TLSTM ####

    if args.forecast_model == "TLSTM":

        ## Log hyperparameters used to train the model
        # general_utils.<method>
        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_params",
            params = {
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "forecast_model": args.forecast_model,
                "lookback_window": args.lookback_window,
                "horizon":args.horizon,
                "model_checkpoint_dir_path": args.model_checkpoint_dir_path,
            }
        )

        torch.manual_seed(args["seed"])

        # instantiate prep
        prep = TLSTMprep(
            datapath   = args.data_path,
            lookback   = args.lookback_window,
            horizon    = args.horizon,
            batch_size = args.batch_size,
            target_col = args.TLSTM.target_col,
            split_date = args.TLSTM.split_date,
            hurdle=args.TLSTM.hurdle,
            delta=args.TLSTM.delta,
            shuffle_train=args.TLSTM.shuffle_train
        )

        # assume prep is your TLSTMprep instance
        Xb, Tb, Yamtb, Yoccb = next(iter(prep.train_loader))
        _, _, n_feat = Xb.shape

        # initialise trainer
        ensemble = TLSTMEnsembleTrainer(
            input_dim   = n_feat,
            hidden_dim  = 64,
            fc_dim      = 32,
            lr_reg      = 1e-3,
            lr_cls      = 1e-3,
            use_compile = True
        )

        # enable matmul
        torch.set_float32_matmul_precision('high')

        # log for device used
        logger.info(f"Using device: {args.device}, for training")

        # Default start epoch
        start_epoch = 1

        # Check for resume flag
        model_checkpoint_path = os.path.join(
            args["model_checkpoint_dir_path"], "Model", "weather.pth"
        )

        if args.resume and os.path.exists(model_checkpoint_path):
            logger.info("Resuming training from checkpoint...")
            ckpt = torch.load(model_checkpoint_path, map_location=args.device)

            ensemble.reg_trainer.model.load_state_dict(  ckpt["regressor_state_dict" ] )
            ensemble.cls_trainer.model.load_state_dict(  ckpt["classifier_state_dict"] )

            start_epoch = ckpt["epoch"] + 1 
            logger.info(f"Resumed from epoch {ckpt['epoch']}. Continuing from epoch {start_epoch}.")

        else:
            logger.info("No checkpoint found. Starting training from scratch.")


        # set checkpoint
        model_checkpoint = math.ceil(args.epochs * 0.1)

        # Run training loop and preserve MLflow logic
        for epoch in range(start_epoch, args.epochs+1):
            logger.info(f"Epoch {epoch}/{args['epochs']} starting...")

            # Train and evaluate for one epoch
            train_loss, train_acc = ensemble.train_one_epoch(
                prep.train_loader,
                val_loader=prep.test_loader
            )
            val_loss = ensemble.val_losses[-1]
            val_acc  = ensemble.val_accuracies[-1]

            # Save checkpoint only every 10 epochs
            if epoch % model_checkpoint == 0 and epoch != 0:
                logger.info("Saving checkpoint at epoch %s.", epoch)

                # Log metric to MLFlow 
                general_utils.mlflow_log(
                    mlflow_init_status,
                    log_function="log_metric",
                    key="train_loss",
                    value=train_loss,
                    step=epoch + step_offset,
                )
                general_utils.mlflow_log(
                    mlflow_init_status,
                    log_function="log_metric",
                    key="train_accuracy",
                    value=train_acc,
                    step=epoch + step_offset,
                )
                general_utils.mlflow_log(
                    mlflow_init_status,
                    log_function="log_metric",
                    key="val_loss",
                    value=val_loss,
                    step=epoch + step_offset,
                )
                general_utils.mlflow_log(
                    mlflow_init_status,
                    log_function="log_metric",
                    key="val_accuracy",
                    value=val_acc,
                    step=epoch + step_offset,
                )

                artifact_subdir = "Model"
                model_checkpoint_path = os.path.join(
                    args["model_checkpoint_dir_path"], artifact_subdir, "weather.pth"
                )
                os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)

                # save both sub-model in one checkpoint
                checkpoint = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    # regressor
                    "regressor_state_dict":   ensemble.reg_trainer.model.state_dict(),
                    # classifier
                    "classifier_state_dict":  ensemble.cls_trainer.model.state_dict(),
                }
                torch.save(checkpoint, model_checkpoint_path)

                ## Use MLflow to log artifact (model checkpoint)
                # general_utils.<method>
                general_utils.mlflow_log(
                    mlflow_init_status, 
                    log_function="log_artifact",
                    local_path=model_checkpoint_path, 
                    artifact_path=artifact_subdir
                )

        ## Use MLflow to log artifact (model config in json)
        # general_utils.<method>
        config_path = os.path.join(args.model_checkpoint_dir_path, "model_config.json")
        with open(config_path, "w") as f:
            json.dump(
                OmegaConf.to_container(args, resolve=True),
                f,
                indent=2
            )

        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_artifact",
            local_path=config_path
        )

        ## Use MLflow to log artifacts (entire `logs`` directory)
        # general_utils.<method>
        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_artifacts",
            local_dir=args.log_dir
        )

        forecast_dir = os.path.join(args.model_checkpoint_dir_path, "forecasts")

        # clear forecasts dir before creating and logging
        general_utils.clear_directory(forecast_dir)

        # Save forecasts and log to MLflow
        forecast_plot = os.path.join(args.model_checkpoint_dir_path, "forecasts/forecastvalidationplot.png")
        ensemble.plot(
            loader=prep.train_loader,
            save_path=forecast_plot
        )

        forecast_plotly = os.path.join(args.model_checkpoint_dir_path, "forecasts/forecastvalidationplot.html")
        metrics = ensemble.plotly(
            loader=prep.train_loader,
            save_path=forecast_plotly
        )

        # Log metrics to MLflow
        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_metrics",
            metrics=metrics
        )

        # Log folder and metadata file as artifacts to MLflow
        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_artifacts",
            local_dir=forecast_dir
        )

        ## Use MLflow to log model for model registry, using pytorch specific methods
        # general_utils.<method>

        # log the regressor
        general_utils.mlflow_pytorch_call(
            mlflow_init_status,
            pytorch_function="log_model",
            pytorch_model=ensemble.reg_trainer.model,
            artifact_path="Model/regressor",
        )

        # log the classifier
        general_utils.mlflow_pytorch_call(
            mlflow_init_status,
            pytorch_function="log_model",
            pytorch_model=ensemble.cls_trainer.model,
            artifact_path="Model/classifier",
        )

#### SARIMA ####

    if args.forecast_model == "SARIMA":

        mlflow.statsmodels.autolog(log_models=False)
        
        ## Log hyperparameters used to train the model
        # general_utils.<method>
        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_params",
            params = {
                "forecast_model": args.forecast_model,
                "pdq": args.pdq,
                "seasonal_pdq": args.seasonal_pdq,
                "model_checkpoint_dir_path": args.model_checkpoint_dir_path,
            }
        )

        # instantiate prep
        prep = SARIMAprep(
            data_path="data/aggregated/rainfall.csv",
            timestamp_col="timestamp",
            split_timestamp="2024-01-01",
            value_col="reading_value_hourly_mean",
            method="ffill",
        )

        train_df, test_df = prep.get_train_test()

        # train and save sarima
        sarimapath = os.path.join(args.model_checkpoint_dir_path, args.forecast_model)
        results, full_pth, gz_pth = fit_and_save_sarimax(
            train_df["reading_value_hourly_mean"],
            tuple(args.pdq),
            tuple(args.seasonal_pdq),
            model_dir=sarimapath
        )
        print("Full model saved to:", full_pth)
        print("Stripped & compressed model saved to:", gz_pth)

        ## Use MLflow to log artifact (model config in json)
        # general_utils.<method>
        config_path = os.path.join(args.model_checkpoint_dir_path, "model_config.json")
        with open(config_path, "w") as f:
            json.dump(
                OmegaConf.to_container(args, resolve=True),
                f,
                indent=2
            )

        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_artifact",
            local_path=config_path
        )

        # log the model
        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_artifact",
            local_path=gz_pth,
            artifact_path="SARIMA"
        )

        ## Use MLflow to log artifacts (entire `logs`` directory)
        # general_utils.<method>
        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_artifacts",
            local_dir=args.log_dir
        )

        forecast_dir = os.path.join(args.model_checkpoint_dir_path, "forecasts")

        # clear forecasts dir before creating and logging
        general_utils.clear_directory(forecast_dir)

        # Save forecasts and log to MLflow
        forecast_plot = os.path.join(args.model_checkpoint_dir_path, "forecasts", "SARIMA", "forecasts.png")
        sarimamodelpath = os.path.join(sarimapath, "model_full.pth")
        metrics = load_and_forecast_SARIMA(
            sarimamodelpath, 
            test_df['reading_value_hourly_mean'],
            y_limit_pct=0.75,
            plot=True,
            plot_path=forecast_plot
            )
        
        # Log metrics to MLflow
        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_metrics",
            metrics=metrics
        )

        # Log folder and metadata file as artifacts to MLflow
        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_artifacts",
            local_dir=forecast_dir
        )

#### RandomForest ####

    if args.forecast_model == "RandomForest":

        mlflow.statsmodels.autolog(log_models=False)
        
        # Log key training hyperparameters to MLflow
        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_params",
            params = {
                "forecast_model": args.forecast_model,
                "model_checkpoint_dir_path": args.model_checkpoint_dir_path,
                "n_estimators": args.RandomForest.n_estimators,
                "random_state": args.RandomForest.random_state,
                "delta": args.RandomForest.delta,
                "horizon": args.RandomForest.horizon,
                "split_date": args.RandomForest.split_date,
            }
        ) 

        # instantiate prep
        prep = RandomForestPrep(
            datapath   = args.data_path,
            horizon    = args.RandomForest.horizon,
            target_col = args.RandomForest.target_col,
            split_date = args.RandomForest.split_date,
            delta=args.RandomForest.delta,
        )

        train_df, test_df = prep.getdfs()

        # train and save random forest
        randomforestpath = os.path.join(args.model_checkpoint_dir_path, args.forecast_model)

        # Train Random Forest
        trainer = RandomForestTrainer(train_df, 
                                      test_df, 
                                      target_col="target", 
                                      n_estimators=args.RandomForest.n_estimators, 
                                      random_state=args.RandomForest.random_state
                                      )
        trainer.train()
        metrics = trainer.evaluate()

        # Log metrics to MLflow
        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_metrics",
            metrics=metrics
        )

        # Ensure the folder exists
        os.makedirs(randomforestpath, exist_ok=True)

        # Save model
        model_pkl_path = os.path.join(randomforestpath, "model.pkl")
        joblib.dump(trainer.get_model(), model_pkl_path)

        ## Use MLflow to log artifact (model config in json)
        # general_utils.<method>
        config_path = os.path.join(args.model_checkpoint_dir_path, "model_config.json")
        with open(config_path, "w") as f:
            json.dump(
                OmegaConf.to_container(args, resolve=True),
                f,
                indent=2
            )

        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_artifact",
            local_path=config_path
        )

        # log the model
        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_artifact",
            local_path=model_pkl_path,
            artifact_path="RandomForest"
        )

        ## Use MLflow to log artifacts (entire `logs`` directory)
        # general_utils.<method>
        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_artifacts",
            local_dir=args.log_dir
        )

        forecast_dir = os.path.join(args.model_checkpoint_dir_path, "forecasts")

        # clear forecasts dir before creating and logging
        general_utils.clear_directory(forecast_dir)

        # Save forecasts and log to MLflow
        html_forecast_path = os.path.join(forecast_dir, "forecast_plot.html")
        trainer.plot_predictions_plotly(html_path=html_forecast_path)

        # Log folder and metadata file as artifacts to MLflow
        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_artifacts",
            local_dir=forecast_dir
        )

#### TimeXer ####

    if args.forecast_model == "TimeXer":
        # Log hyperparameters
        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_params",
            params={
                "forecast_model": args.forecast_model,
                "window_size": args.TimeXer.window_size,
                "prediction_length": args.TimeXer.prediction_length,
                "batch_size": args.batch_size,
                "lr": args.TimeXer.lr,
                "epochs": args.epochs,
                "seed": args.seed,
                "timestamp_col": args.TimeXer.timestamp_col,
                "value_col": args.TimeXer.value_col,
                "feature_cols": args.TimeXer.feature_cols,
                "target_col": args.TimeXer.target_col,
                "num_workers": args.TimeXer.num_workers,
            }
        )

        # Seed
        torch.manual_seed(args.seed)

        # Prepare data
        loader = TimeXerDataLoader(
            datapath = args.data_path,
            timestamp_col = args.TimeXer.timestamp_col,
            value_col = args.TimeXer.value_col,
            feature_cols = args.TimeXer.feature_cols,
            target_col = args.TimeXer.target_col,
            window_size = args.TimeXer.window_size,
            prediction_length = args.TimeXer.prediction_length,
            split_date = pd.Timestamp(args.TimeXer.split_date, tz="Asia/Singapore"),
            batch_size = args.batch_size,
            num_workers = args.TimeXer.num_workers,
        )
        train_loader, val_loader = loader.get_dataloaders()
        df_merged = loader.load_and_merge_dfs()
        df_prepped = loader.prepare_df(df_merged)
        train_ds = loader.make_dataset(df_prepped[df_prepped.index <= loader.split_date])
        test_df = df_prepped[df_prepped.index > loader.split_date]

        # Instantiate your Lightning-wrapped trainer
        trainer_params = {
            "max_epochs":      args.epochs,
            "gradient_clip_val": 0.1,
            # no logger here: we’ll manually pull callback_metrics
        }
        trainer = TimeXerTrainer(
            training_dataset = train_ds,
            train_loader = train_loader,
            val_loader = val_loader,
            test_df = test_df,
            window_size = loader.window_size,
            target_col = loader.target_col,
            trainer_params = trainer_params,
            model_params = None,               # or pass specific model kwargs
            epochs = args.epochs,
        )

        # Lightning will run all epochs internally
        trainer.trainer.fit(trainer.model, train_loader, val_loader)

        # MANUAL METRIC LOGGING (approach 2)
        #    Pull everything Lightning logged into callback_metrics
        metrics = trainer.trainer.callback_metrics
        for key, val in metrics.items():
            # only log scalar Tensors
            if hasattr(val, "item"):
                general_utils.mlflow_log(
                    mlflow_init_status,
                    log_function="log_metric",
                    key=str(key),
                    value=val.item(),
                    step=args.epochs   # or omit step to let MLflow auto-index
                )

        forecast_dir = os.path.join(args.model_checkpoint_dir_path, "forecasts")

        # clear forecasts dir before creating and logging
        general_utils.clear_directory(forecast_dir)

        # Save & log forecast plot
        forecast_txer = os.path.join(args.model_checkpoint_dir_path, "forecasts", "TimeXer")
        os.makedirs(forecast_txer, exist_ok=True)

        png_path = os.path.join(forecast_txer, "forecast.png")
        trainer.plot(save_path=png_path)

        # Log the entire forecasts/TimeXer folder
        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_artifacts",
            local_dir=forecast_txer,
            artifact_path="forecasts/TimeXer"
        )

        # Save & log your resolved config for reproducibility
        cfg_path = os.path.join(args.model_checkpoint_dir_path, "model_config_TimeXer.json")
        with open(cfg_path, "w") as f:
            json.dump(OmegaConf.to_container(args, resolve=True), f, indent=2)
        general_utils.mlflow_log(
            mlflow_init_status,
            log_function="log_artifact",
            local_path=cfg_path
        )

    if mlflow_init_status:
        ## Get artifact link
        artifact_uri = mlflow.get_artifact_uri()
        logger.info("Artifact URI: %s", artifact_uri)
        general_utils.mlflow_log(
            mlflow_init_status, "log_params", params={"artifact_uri": artifact_uri}
        )
        logger.info(
            "Model training with MLflow run ID %s has completed.",
            mlflow_run.info.run_id,
        )
        mlflow.end_run()
    else:
        logger.info("Model training has completed.")

if __name__ == "__main__":
    main()
