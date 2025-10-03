# rainfall-prediction

---

[![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)](#)
[![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?logo=visual-studio-code&logoColor=white)](#)
[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](#)
[![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white)](#)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](#)
[![MySQL](https://img.shields.io/badge/MySQL-4479A1?logo=mysql&logoColor=white)](#)
[![phpMyAdmin](https://img.shields.io/badge/phpMyAdmin-6C78AF?logo=phpmyadmin&logoColor=white)](#)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](#)
[![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)](#)
[![DagsHub](https://img.shields.io/badge/DagsHub-FF6A00?logo=dagsHub&logoColor=white)](#)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-4285F4?logo=googlecloud&logoColor=white)](#)
![Awesome](https://img.shields.io/badge/Awesome-ffd700?logo=awesome&logoColor=black)

---

## Overview
In this project, I am trying to implemente a production ready project using MLOps best practices. We focus on prediction the rainfall using AfriClimate Sensor data in South Africa (Limpopo)


This project is completely setup and run on an AWS EC2 linux machine. Youc can use this [video](https://www.youtube.com/watch?v=1ykg4YmbFVA&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK)
 to setup your remote machine create your postgres database for artifact registory and your S3 bucket. 
## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py



# Setup
## Clone repo

```bash
git clone https://github.com/koomited/rainfall-prediction.git
```
In case you want to try some notebooks: 
```bash
pipenv install ipykernel --dev
pipenv run python -m ipykernel install --user --name=rain-pred

```
## Install pcpu-only torch with pipenv
```bash
PIP_INDEX_URL=https://download.pytorch.org/whl/cpu pipenv install torch
```


## Install the project as local package
```bash
pipenv shell
pip install -e .
```




## Runing mlflow
```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri postgresql://mlflow:PASSWORD@RDS_ENDPOINT_URL:DB_PORT/DB_NAME --default-artifact-root s3://BUCKET_NAME

```
# Run model Training
Make sure you change these files [`config/config.yaml`](config/config.yaml) [`params.yaml`](params.yaml) [`schema.yaml`](schema.yaml),  for your use particulary your mlflow uri in 

```bash
python main.py
```

Then you can access your mlflow ui at [instance_PUBLUC_DNS.com:5000/](http://instance_PUBLUC_DNS.com:5000/) or [localhost:5000](http://localhost:5000)


# Run the flask app with gunicorn

```bash
gunicorn --bind=0.0.0.0:9696 api:app
```

# Deployment
## Deploy the model as web service
In [`deployment/web-service`](deployment/web-service), run in the root directory:
```bash
docker build -t rainfall-prediction-service:v1 .

docker run -d -it --rm -p 9696:9696 rainfall-prediction-service:v1
```

# AWS kinesis for streaming


```bash
#   

KINESIS_STREAM_INPUT=rainfall-events

aws kinesis put-record \
  --stream-name ${KINESIS_STREAM_INPUT} \
  --partition-key 1 \
  --cli-binary-format raw-in-base64-out \
  --data '{
  "info": [
    16.6875, 16.6875, 16.675, 10.35625,
    69.08333333, 198.89583333, 0.51041667, 1.54166667,
    30.18020833, 28.31041667, 14.8875, 19.86875,
    21.10833333, 17.51458333, 17.48333333, 18.06666667,
    17.83958333, 14.8875, 19.86875, 21.09583333,
    17.51458333, 17.48333333, 18.0625, 17.83958333,
    15.04375, 18.68333333, 20.34375, 16.93125,
    16.50625, 17.62916667, 18.16458333, 10.95,
    7.45416667, 8.72083333, 8.59791667, 6.60833333,
    8.78125, 11.77708333, 77.89583333, 46.02083333,
    48.35416667, 58.64583333, 52.54166667, 58.4375,
    69.52083333, 273.35416667, 167.33333333, 236.58333333,
    133.10416667, 181.66666667, 213.0625, 92.875,
    0.48125, 2.03958333, 1.57291667, 0.84166667,
    0.60625, 0.56458333, 0.76458333, 1.625,
    5.2875, 3.70208333, 2.08541667, 1.58333333,
    1.65208333, 2.18958333, 30.18270833, 30.00541667,
    29.84958333, 29.98916667, 30.05708333, 30.02583333,
    30.14666667, 28.31354167, 28.136875, 27.980625,
    28.12041667, 28.18791667, 28.15729167, 28.2775,
    0.8, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0
  ],
  "event_id": 125
}
'
```

## Kinesis event
```bash
{
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49667636532760164205621325266490096894654679758160265218",
                "data": "ewogICJpbmZvIjogWwogICAgMTYuNjg3NSwgMTYuNjg3NSwgMTYuNjc1LCAxMC4zNTYyNSwKICAgIDY5LjA4MzMzMzMzLCAxOTguODk1ODMzMzMsIDAuNTEwNDE2NjcsIDEuNTQxNjY2NjcsCiAgICAzMC4xODAyMDgzMywgMjguMzEwNDE2NjcsIDE0Ljg4NzUsIDE5Ljg2ODc1LAogICAgMjEuMTA4MzMzMzMsIDE3LjUxNDU4MzMzLCAxNy40ODMzMzMzMywgMTguMDY2NjY2NjcsCiAgICAxNy44Mzk1ODMzMywgMTQuODg3NSwgMTkuODY4NzUsIDIxLjA5NTgzMzMzLAogICAgMTcuNTE0NTgzMzMsIDE3LjQ4MzMzMzMzLCAxOC4wNjI1LCAxNy44Mzk1ODMzMywKICAgIDE1LjA0Mzc1LCAxOC42ODMzMzMzMywgMjAuMzQzNzUsIDE2LjkzMTI1LAogICAgMTYuNTA2MjUsIDE3LjYyOTE2NjY3LCAxOC4xNjQ1ODMzMywgMTAuOTUsCiAgICA3LjQ1NDE2NjY3LCA4LjcyMDgzMzMzLCA4LjU5NzkxNjY3LCA2LjYwODMzMzMzLAogICAgOC43ODEyNSwgMTEuNzc3MDgzMzMsIDc3Ljg5NTgzMzMzLCA0Ni4wMjA4MzMzMywKICAgIDQ4LjM1NDE2NjY3LCA1OC42NDU4MzMzMywgNTIuNTQxNjY2NjcsIDU4LjQzNzUsCiAgICA2OS41MjA4MzMzMywgMjczLjM1NDE2NjY3LCAxNjcuMzMzMzMzMzMsIDIzNi41ODMzMzMzMywKICAgIDEzMy4xMDQxNjY2NywgMTgxLjY2NjY2NjY3LCAyMTMuMDYyNSwgOTIuODc1LAogICAgMC40ODEyNSwgMi4wMzk1ODMzMywgMS41NzI5MTY2NywgMC44NDE2NjY2NywKICAgIDAuNjA2MjUsIDAuNTY0NTgzMzMsIDAuNzY0NTgzMzMsIDEuNjI1LAogICAgNS4yODc1LCAzLjcwMjA4MzMzLCAyLjA4NTQxNjY3LCAxLjU4MzMzMzMzLAogICAgMS42NTIwODMzMywgMi4xODk1ODMzMywgMzAuMTgyNzA4MzMsIDMwLjAwNTQxNjY3LAogICAgMjkuODQ5NTgzMzMsIDI5Ljk4OTE2NjY3LCAzMC4wNTcwODMzMywgMzAuMDI1ODMzMzMsCiAgICAzMC4xNDY2NjY2NywgMjguMzEzNTQxNjcsIDI4LjEzNjg3NSwgMjcuOTgwNjI1LAogICAgMjguMTIwNDE2NjcsIDI4LjE4NzkxNjY3LCAyOC4xNTcyOTE2NywgMjguMjc3NSwKICAgIDAuOCwgMC4wLCAwLjAsIDAuMCwKICAgIDAuMCwgMC4wLCAwLjAKICBdLAogICJldmVudF9pZCI6IDEyMwp9Cg==",
                "approximateArrivalTimestamp": 1759418864.745
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49667636532760164205621325266490096894654679758160265218",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::541690257764:role/lambda-kinesis-role",
            "awsRegion": "us-east-1",
            "eventSourceARN": "arn:aws:kinesis:us-east-1:541690257764:stream/rainfall-events"
        }
    ]
}
```

## Reag from a stream
```bash
KINESIS_STREAM_OUTPUT='rainfall-predictions'
SHARD='shardId-000000000000'

SHARD_ITERATOR=$(aws kinesis \
    get-shard-iterator \
        --shard-id ${SHARD} \
        --shard-iterator-type TRIM_HORIZON \
        --stream-name ${KINESIS_STREAM_OUTPUT} \
        --query 'ShardIterator' \
)

RESULT=$(aws kinesis get-records --shard-iterator $SHARD_ITERATOR)

echo ${RESULT} | jq -r '.Records[0].Data' | base64 --decode | jq
```

```bash

export PREDICTION_STREAM_NAME='rainfall-predictions'
export RUN_ID="8de0cb304e844db8ae045f16c26c71db"
export TEST_RUN="True"

python test3.py
```

```bash
docker build -t stream-model-rainfall:v1 -f lambda_dockerfile .

docker run -d -it --rm \
    -p 8080:8080 \
    -e PREDICTION_STREAM_NAME='rainfall-predictions' \
    -e  RUN_ID="8de0cb304e844db8ae045f16c26c71db" \
    -e TEST_RUN="True" \
    -e AWS_DEFAULT_REGION="us-east-1" \
    stream-model-rainfall:v1
```

## Publish lambda function image to ECR


```bash
aws ecr create-repository --repository-name rainfall-model

```
Logging in

```bash
aws ecr get-login-password --region us-east-1 \
  | docker login \
    --username AWS \
    --password-stdin 541690257764.dkr.ecr.us-east-1.amazonaws.com
```


```bash
REMOTE_URI="541690257764.dkr.ecr.us-east-1.amazonaws.com/rainfall-model"
REMOTE_TAG="v1"
REMOTE_IMAGE=${REMOTE_URI}:${REMOTE_TAG}

LOCAL_IMAGE="stream-model-rainfall:v1"
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE}
docker push ${REMOTE_IMAGE}
```

# Unit testing

```bash 
docker build -t stream-model-rainfall:v2 -f lambda_dockerfile .

docker run -d -it --rm \
    -p 8080:8080 \
    -e PREDICTION_STREAM_NAME='rainfall-predictions' \
    -e  RUN_ID="8de0cb304e844db8ae045f16c26c71db" \
    -e TEST_RUN="True" \
    -e AWS_DEFAULT_REGION="us-east-1" \
    stream-model-rainfall:v2
```