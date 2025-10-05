# pylint: disable=duplicate-code
import json

import requests
from deepdiff import DeepDiff

with open('event.json', 'rt', encoding='utf-8') as f_in:
    event = json.load(f_in)


url = "http://localhost:8080/2015-03-31/functions/function/invocations"
actual_response = requests.post(url, json=event, timeout=20).json()
print("Actual response:")

print(json.dumps(actual_response, indent=2))

expected_response = expected_prediction = {
    "predictions": [
        {
            'model': 'rainfall-prediction',
            "version": "8de0cb304e844db8ae045f16c26c71db",
            "prediction": {
                'rainfall': 0,
                'rainfall_history_id': actual_response["predictions"][0]["prediction"][
                    "rainfall_history_id"
                ],
            },
        }
    ]
}

diff = DeepDiff(actual_response, expected_response, significant_digits=3)
print('diff=', diff)

assert 'type_changes' not in diff
assert 'values_changed' not in diff

assert actual_response == expected_response
