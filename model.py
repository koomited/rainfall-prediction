import os
import json
import uuid
import base64

import boto3
import numpy as np

from mlProject.pipeline.prediction import PredictionPipeline

# REGION_NAME = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
# kinesis_client = boto3.client(
#     "kinesis",
#     region_name=REGION_NAME  # replace with your stream's region
# )


def base64_decode(encoded_data):
    decoded_data = base64.b64decode(encoded_data).decode('utf-8')
    rainfall_event = json.loads(decoded_data)
    return rainfall_event


class ModelService:
    def __init__(self, model, model_version, callbacks=None):
        self.model = model
        self.model_version = model_version
        if callbacks is None:
            self.callbacks = []
        else:
            self.callbacks = callbacks or []

    def prepare_feature(self, data):
        return np.array(data).reshape(1, -1)

    def predict(self, data):
        return self.model.predict(data)

    def lambda_handler(self, event):

        # print(json.dumps(event))
        predictions_events = []
        for record in event['Records']:
            encoded_data = record['kinesis']['data']
            rainfall_event = base64_decode(encoded_data)

            rainfall_history_event = rainfall_event['info']
            rainfall_history_id = str(uuid.uuid4())

            rainfall_history_event = self.prepare_feature(rainfall_history_event)

            prediction = self.predict(rainfall_history_event)

            prediction_event = {
                'model': 'rainfall-prediction',
                'version': self.model_version,
                'prediction': {
                    'rainfall': prediction,
                    'rainfall_history_id': rainfall_history_id,
                },
            }

            # if not TEST_RUN:
            #     kinesis_client.put_record(
            #         StreamName=PREDICTION_STREAM_NAME,
            #         Data=json.dumps(prediction_event),
            #         PartitionKey=str(rainfall_history_id)
            #     )

            predictions_events.append(prediction_event)

        for callback in self.callbacks:
            callback(predictions_events)
        return {'predictions': predictions_events}


class KinesisCallback:

    def __init__(self, kinesis_client, prediction_stream_name):
        self.kinesis_client = kinesis_client
        self.prediction_stream_name = prediction_stream_name

    def put_record(self, prediction_events):
        for prediction_event in prediction_events:
            ride_id = prediction_event["prediction"]["rainfall_history_id"]

            self.kinesis_client.put_record(
                StreamName=self.prediction_stream_name,
                Data=json.dumps(prediction_event),
                PartitionKey=str(ride_id),
            )


def create_kinesis_client():
    """
    Create a kinesis client
    """
    endpoint_url = os.getenv("KINESIS_ENDPOINT_URL")
    if endpoint_url is None:
        return boto3.client("kinesis", region_name="us-east-1")
    return boto3.client("kinesis", region_name="us-east-1", endpoint_url=endpoint_url)


def init(prediction_stream_name: str, run_id: str, test_run: bool):

    model = PredictionPipeline(run_id)

    callbacks = []

    if not test_run:

        kinesis_client = create_kinesis_client()

        kinesis_callbacks = KinesisCallback(kinesis_client, prediction_stream_name)
        callbacks.append(kinesis_callbacks.put_record)

    model_service = ModelService(model=model, model_version=run_id, callbacks=callbacks)

    return model_service
