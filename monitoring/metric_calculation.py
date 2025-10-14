import datetime
import time
import random
import logging
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib
from evidently import ColumnMapping
from evidently.report import Report
from prefect import task, flow
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from mlProject.pipeline.prediction import PredictionPipeline
import pandas as pd
from copy import deepcopy as dc
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


pd.set_option('future.no_silent_downcasting', True)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists metrics;
create table metrics(
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float
)

"""

reference_data = pd.read_parquet("data/reference.parquet")
raw_data = pd.read_parquet("data/train.parquet")
RUN_ID = os.getenv("RUN_ID", "8de0cb304e844db8ae045f16c26c71db")


class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)
        def     __getitem__(self, i):
            return self.X[i], self.y[i]
        

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
    
    
    df_results = dc(X_df)
    df_results.loc[:, "rainfall_actual"] = y_test
    df_results.loc[:, "rainfall_predicted"] = predictions
    df_results.loc[:, "date"] =pd.to_datetime(df_results.index)
    
    
    return df_results



begin = datetime.datetime(2025, 6, 29, 0, 0)

# num_features = ['passenger_count', 'trip_distance','fare_amount', 'total_amount']
# cat_features = ['PULocationID', 'DOLocationID']

# column_mapping = ColumnMapping(
#     target=None,
#     prediction="prediction",
#     numerical_features=num_features,
#     categorical_features=cat_features
# )

# report = Report(
#     metrics = [
#         ColumnDriftMetric(
#             column_name="prediction"
#         ),
#         DatasetDriftMetric(),
#         DatasetMissingValuesMetric()
        
#     ]
# )

@task
def prep_db():
    with psycopg.connect(
        "host=localhost port=5432 user=postgres password=example",autocommit=True
        ) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall())==0:
            conn.execute('create database test;')

    with psycopg.connect(
        "host=localhost port=5432 dbname=test user=postgres password=example",autocommit=True
    ) as conn:
        conn.execute(create_table_statement)


@task
def calculate_metrics(curr, i):
    current_data = raw_data[(raw_data.date<=(begin+datetime.timedelta(i)))&
                           (raw_data.date<=(begin+datetime.timedelta(i+1)))]
    
    current_data.fillna(0, inplace=True)
    current_data["prediction"] = model.predict(current_data[num_features+cat_features])
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    result = report.as_dict()
    
    # prediction drift
    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    # Number of driftet column
    num_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]
    share_missing_values = result["metrics"][2]["result"]["current"]["share_of_missing_values"]
    
    curr.execute(
        "INSERT INTO metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values(%s, %s, %s, %s)",
        (datetime.datetime.now(pytz.timezone("Africa/Porto-Novo")
), prediction_drift, num_drifted_columns, share_missing_values)
    )

@flow    
def batch_monitoring_backfill():
    prep_db()
    last_send = datetime.datetime.now()-datetime.timedelta(seconds=10)
    with psycopg.connect(
        "host=localhost port=5432 dbname=test user=postgres password=example",autocommit=True
        ) as conn:
        for i in range(0, 27):
            with conn.cursor() as curr:
                calculate_metrics(curr, i)
            
            new_send = datetime.datetime.now()
            second_elapsed = (new_send-last_send).total_seconds()
            if second_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT-second_elapsed)
            while last_send< new_send:
                last_send= last_send + datetime.timedelta(seconds=10)
            logging.info("Data Sent")
            

    
if __name__=="__main__":
    batch_monitoring_backfill()