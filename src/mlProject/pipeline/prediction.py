import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

class PredictionPipeline:
    def __init__(self):
        self.classifier = joblib.load(Path(BASE_DIR / 'artifacts/model_trainer/classification.joblib'))
        self.scaler = joblib.load(Path(BASE_DIR / 'artifacts/model_trainer/scaler.joblib'))
        self.regressor = torch.load(Path(BASE_DIR / 'artifacts/model_trainer/lstm.pth'), weights_only=False)
        

    
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