import numpy as np
from flask import Flask, jsonify, request

from mlProject.pipeline.prediction import RUN_ID, PredictionPipeline
import os

app = Flask("rainfall-prediction")
RUN_ID = os.getenv('RUN_ID', "8de0cb304e844db8ae045f16c26c71db")


@app.route("/predict", methods=["POST"])
def prediction_endpoint():
    rain_info = request.get_json()
    predict_object = PredictionPipeline(run_id=RUN_ID)
    data = rain_info["info"]
    data = np.array(data).reshape(1, -1)
    pred = predict_object.predict(data)
    print("prediction: ", pred)

    result = {"rainfall": pred, "model_version": RUN_ID}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
