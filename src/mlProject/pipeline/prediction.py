import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from pathlib import Path
import joblib
import mlflow
import os

RUN_ID = os.getenv('RUN_ID', "8de0cb304e844db8ae045f16c26c71db")


class PredictionPipeline:
    def __init__(self, run_id):
        
        self.S3_BASE_URL = f"s3://koomi-mlflow-artifacts-remote/3/{run_id}/artifacts/"
        
        self.classifier = joblib.load(mlflow.artifacts.download_artifacts(f"{self.S3_BASE_URL}classifier/classification.joblib"))
        self.scaler = joblib.load(mlflow.artifacts.download_artifacts(f"{self.S3_BASE_URL}scaler/scaler.joblib"))
        self.regressor = torch.load(mlflow.artifacts.download_artifacts(f"{self.S3_BASE_URL}regressor/lstm.pth"), weights_only=False)
        
    
    def predict(self, data):

        scaled_data = self.scaler.transform(data)
        class_prediction = self.classifier.predict(scaled_data).item()
        
        if class_prediction == 0:
            pred = 0
        else:
            device = "cpu"

            model = self.regressor.to(device)
            model.eval()
            lstm_data = torch.tensor(scaled_data).float()
            pred = model(lstm_data.unsqueeze(0))
            
        return pred