import os
from copy import deepcopy as dc
from urllib.parse import urlparse

import numpy as np
import torch
import joblib
import mlflow
import pandas as pd
import torch.nn as nn
import mlflow.sklearn
from torch import optim
from sklearn.metrics import root_mean_squared_error
from torch.utils.data import Dataset, DataLoader

from mlProject import logger
from mlProject.constants import *
from mlProject.utils.common import read_yaml, save_json, create_directories
from mlProject.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    def log_into_mlflow(self):

        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("rainfall-prediction")
        with mlflow.start_run():
            test_data = pd.read_csv(self.config.test_data_path, index_col="time")

            classification_target_column = list(
                self.config.classification_target_column.keys()
            )[0]
            regression_target_column = list(
                self.config.regression_target_column.keys()
            )[0]
            targets_columns = [classification_target_column, regression_target_column]

            # Features
            X_test = test_data.drop(columns=targets_columns).values

            # Targets
            y_test = test_data[regression_target_column].values.reshape(-1, 1)

            # Load classifier + scaler
            scaler = joblib.load(self.config.scaler_path)
            classifier = joblib.load(self.config.classifier_path)

            # Scale features
            X_test_scaled = scaler.transform(X_test)
            classifier_preds = classifier.predict(X_test_scaled)

            # Prepare LSTM input
            X_test_lstm = dc(np.flip(X_test_scaled, axis=1))
            vars_dim = X_test_lstm.shape[1]
            X_test_lstm = X_test_lstm.reshape((-1, vars_dim, 1))

            # Torch tensors
            X_test_tensor = torch.tensor(X_test_lstm).float()
            y_test_tensor = torch.tensor(y_test).float()

            test_dataset = self.TimeSeriesDataset(X_test_tensor, y_test_tensor)
            batch_size = int(self.config.batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load regression model
            model = torch.load(self.config.regressor_path, weights_only=False)
            model.to(device)
            model.eval()

            predictions = []
            with torch.no_grad():
                for i, (X_batch, _) in enumerate(test_loader):
                    X_batch = X_batch.to(device)

                    # If classifier said 0 → output 0
                    clf_batch_preds = classifier_preds[
                        i * batch_size : (i + 1) * batch_size
                    ]
                    batch_preds = []
                    if len(clf_batch_preds) < len(X_batch):
                        # Handle last smaller batch
                        X_batch = X_batch[: len(clf_batch_preds)]

                    for j, clf_pred in enumerate(clf_batch_preds):
                        if clf_pred == 0:
                            batch_preds.append(0.0)
                        else:
                            reg_out = model(X_batch[j].unsqueeze(0))
                            batch_preds.append(reg_out[0].item())

                    predictions.extend(batch_preds)

            predictions = np.array(predictions).reshape(-1, 1)

            scaler_path = self.config.scaler_path
            regressor_path = self.config.regressor_path
            classifier_path = self.config.classifier_path

            rmse = root_mean_squared_error(y_test, predictions)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_params(self.config.classifier_params)
            mlflow.log_params(self.config.regressor_params)
            mlflow.log_artifact(scaler_path, artifact_path="scaler")
            mlflow.log_artifact(classifier_path, artifact_path="classifier")
            mlflow.log_artifact(regressor_path, artifact_path="regressor")
            mlflow.log_metric('rmse', rmse)

            logger.info(f"Final RMSE: {rmse}")
            score = {"rmse": rmse}
            save_json(path=Path(self.config.metric_file_name), data=score)
