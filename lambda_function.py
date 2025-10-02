import json
import base64
import boto3
import os
import numpy as np
from mlProject.pipeline.prediction import PredictionPipeline

predictor = PredictionPipeline()

REGION_NAME = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
kinesis_client = boto3.client(
    "kinesis",
    region_name=REGION_NAME  # replace with your stream's region
)


PREDICTION_STREAM_NAME = os.getenv('PREDICTION_STREAM_NAME', 'rainfall-predictions')
RUN_ID = os.getenv('RUN_ID')
TEST_RUN = os.getenv("TEST_RUN", "False") == "True"


def predict(data):
    return 0.0

def lambda_handler(event, context):
    
    
    # print(json.dumps(event))
    predictions_events = []
    for record in event['Records']:
        encoded_data = record['kinesis']['data']
        data_decoded = base64.b64decode(encoded_data).decode('utf-8')
        rainfall_event = json.loads(data_decoded)


        rainfall_history_event = rainfall_event['info']
        rainfall_history_id = rainfall_event['event_id']
        
        rainfall_history_event = np.array(rainfall_history_event).reshape(1,-1)
        
        
        prediction = predictor.predict(rainfall_history_event)

        prediction_event = {
            'model': 'rainfall-prediction',
            'version': '1.0',
            'prediction':{
                'rainfall': prediction,
                'rainfall_history_id': rainfall_history_id
                
            }
        }
        if not TEST_RUN:
            kinesis_client.put_record(
                StreamName=PREDICTION_STREAM_NAME,
                Data=json.dumps(prediction_event),
                PartitionKey=str(rainfall_history_id)
            )

        predictions_events.append(prediction_event)

    return {
        'predictions': predictions_events
    }

