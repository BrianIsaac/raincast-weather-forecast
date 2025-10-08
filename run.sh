#!/bin/bash

# Description:
# This script runs train.py with user-provided MLflow credentials

echo "Enter your MLflow account (username or email):"
read MLFLOW_ACCOUNT

echo "Enter your MLflow password:"
read -s MLFLOW_PASSWORD

# Export environment variables for use in train.py or MLflow authentication
export MLFLOW_TRACKING_USERNAME=$MLFLOW_ACCOUNT
export MLFLOW_TRACKING_PASSWORD=$MLFLOW_PASSWORD

echo "MLflow credentials set."

# Change directory to src and run train.py
echo "Running train.py..."
python src/train.py

echo "Training completed."