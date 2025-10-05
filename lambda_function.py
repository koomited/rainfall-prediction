import os

import model

PREDICTION_STREAM_NAME = os.getenv('PREDICTION_STREAM_NAME', 'rainfall-predictions')
RUN_ID = os.getenv('RUN_ID', "8de0cb304e844db8ae045f16c26c71db")

TEST_RUN = os.getenv("TEST_RUN", "False") == "True"


model_service = model.init(
    prediction_stream_name=PREDICTION_STREAM_NAME, run_id=RUN_ID, test_run=TEST_RUN
)


def lambda_handler(event, context):

    return model_service.lambda_handler(event)
