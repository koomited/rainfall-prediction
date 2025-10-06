LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
LOCAL_IMAGE_NAME="stream-model-duration:${LOCAL_TAG}"

test:
	pytest tests/

quality_check:
	isort .
	black .
	pylint --recursive=y . || true


build: test quality_check
	docker build -t ${LOCAL_IMAGE_NAME} .

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash ./integration-test/run.sh

publish: build integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash script/publish.sh

setup: 
	pipenv install --dev 
	pre-commit install
	
all: setup publish
