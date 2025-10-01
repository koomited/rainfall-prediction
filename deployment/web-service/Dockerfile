FROM python:3.12-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY . ./

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "deployment/web-service/predict:app"]