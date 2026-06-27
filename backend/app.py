from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .services import (
    ASSET_DIR,
    find_twins,
    get_dashboard_payload,
    predict_patient,
    simulate_patient,
    top_modifiable_drivers,
)

app = Flask(__name__)


def _json_body() -> dict:
    """Parse the request JSON, tolerating empty/malformed bodies."""
    try:
        body = request.get_json(force=True, silent=True)
    except Exception:  # noqa: BLE001 - any parse failure becomes an empty body
        body = None
    return body if isinstance(body, dict) else {}


def _as_inputs(value: object) -> dict:
    return value if isinstance(value, dict) else {}


@app.errorhandler(ValueError)
def _handle_value_error(err: ValueError):
    return jsonify({"error": "Invalid request", "detail": str(err)}), 400


@app.get("/api/dashboard")
def dashboard():
    return jsonify(get_dashboard_payload())


@app.post("/api/predict")
def predict():
    body = _json_body()
    return jsonify(predict_patient(_as_inputs(body.get("inputs")), body.get("patientRef")))


@app.post("/api/simulate")
def simulate():
    body = _json_body()
    return jsonify(
        simulate_patient(
            body.get("patientRef"),
            _as_inputs(body.get("modifiedInputs")),
            body.get("baselineInputs"),
        )
    )


@app.post("/api/drivers")
def drivers():
    body = _json_body()
    return jsonify(
        top_modifiable_drivers(
            body.get("patientRef"),
            body.get("baselineInputs"),
        )
    )


@app.post("/api/twins")
def twins():
    body = _json_body()
    return jsonify(
        find_twins(
            body.get("patientRef"),
            _as_inputs(body.get("inputs")),
            body.get("n", 12),
        )
    )


@app.get("/api/assets/<path:asset_path>")
def assets(asset_path: str):
    directory = Path(ASSET_DIR)
    return send_from_directory(directory, asset_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
