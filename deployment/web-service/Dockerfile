FROM python:3.12-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR app

COPY . app


# Free up space and install torch directly
RUN pipenv install --deploy --system \
    && RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /root/.cache/pip




EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "deployment/web-service/predict:app"]