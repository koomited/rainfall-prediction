import numpy as np
from flask import Flask, jsonify, request

from mlProject.pipeline.prediction import RUN_ID, PredictionPipeline

app = Flask("rainfall-prediction")


@app.route("/predict", methods=["POST"])
def prediction_endpoint():
    rain_info = request.get_json()
    predict_object = PredictionPipeline()
    data = rain_info["info"]
    data = np.array(data).reshape(1, -1)
    pred = predict_object.predict(data)
    print("prediction: ", pred)

    result = {"rainfall": pred, "model_version": RUN_ID}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
