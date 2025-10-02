FROM python:3.12-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR app

COPY . ./


# Free up space and install torch directly
RUN pipenv install --deploy --system \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e . \
    && rm -rf /root/.cache/pip





EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "api:app"]