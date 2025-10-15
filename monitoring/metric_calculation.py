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
create table if not exists metrics(
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
        

def predict(X_df, y, run_id):
    model_object = PredictionPipeline(run_id=run_id)
    
    y = y.values.reshape(-1, 1)

    # Scale features
    X_test = X_df.values
    X_test_scaled = model_object.scaler.transform(X_test)
    classifier_preds = model_object.classifier.predict(X_test_scaled)

    # Prepare LSTM input
    X_test_lstm = dc(np.flip(X_test_scaled, axis=1))
    vars_dim = X_test_lstm.shape[1]
    X_test_lstm = X_test_lstm.reshape((-1, vars_dim, 1))
    X_test_tensor = torch.tensor(X_test_lstm).float()
    y_tensor = torch.tensor(y).float()
    
    test_dataset = TimeSeriesDataset(X_test_tensor, y_tensor)
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
    df_results.loc[:, "rainfall_actual"] = y
    df_results.loc[:, "rainfall_predicted"] = predictions
    df_results.loc[:, "date"] =pd.to_datetime(df_results.index)
    
    
    return df_results


num_features = ['temperature', 'feels_like', 'app_temp', 'dew_point', 'humidity',
       'wind_direction', 'wind_speed', 'wind_gust', 'pressure_relative',
       'pressure_absolute', 'temperature(t-1)', 'temperature(t-2)',
       'temperature(t-3)', 'temperature(t-4)', 'temperature(t-5)',
       'temperature(t-6)', 'temperature(t-7)', 'feels_like(t-1)',
       'feels_like(t-2)', 'feels_like(t-3)', 'feels_like(t-4)',
       'feels_like(t-5)', 'feels_like(t-6)', 'feels_like(t-7)',
       'app_temp(t-1)', 'app_temp(t-2)', 'app_temp(t-3)', 'app_temp(t-4)',
       'app_temp(t-5)', 'app_temp(t-6)', 'app_temp(t-7)', 'dew_point(t-1)',
       'dew_point(t-2)', 'dew_point(t-3)', 'dew_point(t-4)', 'dew_point(t-5)',
       'dew_point(t-6)', 'dew_point(t-7)', 'humidity(t-1)', 'humidity(t-2)',
       'humidity(t-3)', 'humidity(t-4)', 'humidity(t-5)', 'humidity(t-6)',
       'humidity(t-7)', 'wind_direction(t-1)', 'wind_direction(t-2)',
       'wind_direction(t-3)', 'wind_direction(t-4)', 'wind_direction(t-5)',
       'wind_direction(t-6)', 'wind_direction(t-7)', 'wind_speed(t-1)',
       'wind_speed(t-2)', 'wind_speed(t-3)', 'wind_speed(t-4)',
       'wind_speed(t-5)', 'wind_speed(t-6)', 'wind_speed(t-7)',
       'wind_gust(t-1)', 'wind_gust(t-2)', 'wind_gust(t-3)', 'wind_gust(t-4)',
       'wind_gust(t-5)', 'wind_gust(t-6)', 'wind_gust(t-7)',
       'pressure_relative(t-1)', 'pressure_relative(t-2)',
       'pressure_relative(t-3)', 'pressure_relative(t-4)',
       'pressure_relative(t-5)', 'pressure_relative(t-6)',
       'pressure_relative(t-7)', 'pressure_absolute(t-1)',
       'pressure_absolute(t-2)', 'pressure_absolute(t-3)',
       'pressure_absolute(t-4)', 'pressure_absolute(t-5)',
       'pressure_absolute(t-6)', 'pressure_absolute(t-7)', 'rainfall(t-1)',
       'rainfall(t-2)', 'rainfall(t-3)', 'rainfall(t-4)', 'rainfall(t-5)',
       'rainfall(t-6)', 'rainfall(t-7)']
begin = datetime.datetime(2025, 6, 29, 0, 0)

column_mapping = ColumnMapping(
    target=None,
    prediction="rainfall_predicted",
    numerical_features=num_features
)

report = Report(
    metrics = [
        ColumnDriftMetric(
            column_name="rainfall_predicted"
        ),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
        
    ]
)




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

from prefect.cache_policies import NO_CACHE

@task(cache_policy=NO_CACHE)
def calculate_metrics(curr, i):
    current_data = raw_data[(raw_data.date>=(begin+datetime.timedelta(days = i)))&
                           (raw_data.date<=(begin+datetime.timedelta(days = i+1)))]
    
    current_data= predict(current_data[num_features],
                            y=current_data.rainfall_actual, 
                            run_id=RUN_ID)
    report.run(reference_data=reference_data.reset_index(drop=True), 
               current_data=current_data.reset_index(drop=True), 
               column_mapping=column_mapping)
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
        for i in range(0, 27, 8):
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