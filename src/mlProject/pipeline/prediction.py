import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
import torch


class PredictionPipeline:
    def __init__(self):
        self.classifier = joblib.load(Path('artifacts/model_trainer/classification.joblib'))
        self.scaler = joblib.load(Path('artifacts/model_trainer/scaler.joblib'))
        self.regressor = torch.load(Path('artifacts/model_trainer/lstm.pth'), weights_only=False)
        

    
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