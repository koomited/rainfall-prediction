#!/usr/bin/env python
# coding: utf-8
import os
from mlProject.pipeline.prediction import PredictionPipeline
import pandas as pd
from copy import deepcopy as dc
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import uuid

import sys








def read_data(filename: str):
    data = pd.read_csv(filename, index_col="time")
    features = data.drop(columns=["is_rain", "rainfall"])
    
    target = data["rainfall"].values
    
    return features, target
    
    
    




# In[8]:


class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)
        def     __getitem__(self, i):
            return self.X[i], self.y[i]
        



def save_results(results, output_file):
    results.to_parquet(output_file)


# In[10]:


def predict(X_df, y_test, run_id, output_file):
    model_object = PredictionPipeline(run_id=run_id)
    
    y_test = y_test.reshape(-1, 1)

    # Scale features
    X_test = X_df.values
    X_test_scaled = model_object.scaler.transform(X_test)
    classifier_preds = model_object.classifier.predict(X_test_scaled)

    # Prepare LSTM input
    X_test_lstm = dc(np.flip(X_test_scaled, axis=1))
    vars_dim = X_test_lstm.shape[1]
    X_test_lstm = X_test_lstm.reshape((-1, vars_dim, 1))
    X_test_tensor = torch.tensor(X_test_lstm).float()
    y_test_tensor = torch.tensor(y_test).float()
    
    test_dataset = TimeSeriesDataset(X_test_tensor, y_test_tensor)
    batch_size = 8
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model_object.regressor
    
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for i, (X_batch, _) in enumerate(test_loader):
            X_batch = X_batch.to(device)
            
            # If classifier said 0 → output 0
            clf_batch_preds = classifier_preds[i * batch_size : (i + 1) * batch_size]
            batch_preds = []
            if len(clf_batch_preds) < len(X_batch):
                # Handle last smaller batch
                X_batch = X_batch[:len(clf_batch_preds)]
            
            for j, clf_pred in enumerate(clf_batch_preds):
                if clf_pred == 0:
                    batch_preds.append(0.0)
                else:
                    reg_out = model(X_batch[j].unsqueeze(0))
                    batch_preds.append(reg_out[0].item())
            
            predictions.extend(batch_preds)

    predictions = np.array(predictions).reshape(-1, 1)
    
    n = len(X_df)
    rainfall_ids = []
    for _ in range(n):
        rainfall_ids.append(str(uuid.uuid4()))  
        
    
    df_results = dc(X_df)
    df_results.insert(0, "rainfall_id", rainfall_ids)
    df_results.loc[:, "rainfall_actual"] = y_test
    df_results.loc[:, "rainfall_predicted"] = predictions
    df_results.loc[:, "diff"] = df_results["rainfall_actual"] - df_results["rainfall_predicted"]
    df_results.loc[:, "model_version"] = run_id
    
    
    save_results(df_results, output_file)


def run():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    RUN_ID = sys.argv[3]
    
    os.chdir("../")
    os.system('mkdir batch_deploy/output')
    # input_file = "artifacts/data_transformation/test.csv"
    # output_file = "batch_deploy/output/rainfall_preds.parquet"
    # RUN_ID = os.getenv("RUN_ID", "8de0cb304e844db8ae045f16c26c71db")
    X_test, y_test = read_data(input_file)
    predict(X_df=X_test, y_test=y_test, run_id=RUN_ID, output_file=output_file)
    



if __name__ == "__main__":
    run()
    