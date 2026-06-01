from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .services import (
    ASSET_DIR,
    get_dashboard_payload,
    predict_patient,
    simulate_patient,
    top_modifiable_drivers,
)

app = Flask(__name__)


@app.get("/api/dashboard")
def dashboard():
    return jsonify(get_dashboard_payload())


@app.post("/api/predict")
def predict():
    body = request.get_json(force=True) or {}
    return jsonify(predict_patient(body.get("inputs", {}), body.get("patientRef")))


@app.post("/api/simulate")
def simulate():
    body = request.get_json(force=True) or {}
    return jsonify(
        simulate_patient(
            body.get("patientRef"),
            body.get("modifiedInputs", {}),
            body.get("baselineInputs"),
        )
    )


@app.post("/api/drivers")
def drivers():
    body = request.get_json(force=True) or {}
    return jsonify(
        top_modifiable_drivers(
            body.get("patientRef"),
            body.get("baselineInputs"),
        )
    )


@app.get("/api/assets/<path:asset_path>")
def assets(asset_path: str):
    directory = Path(ASSET_DIR)
    return send_from_directory(directory, asset_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
