from __future__ import annotations

import json
import math
import pickle
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
ASSET_DIR = ROOT / "assets"

MODULES = [
    "Cohort Overview",
    "Patient Risk Assessment",
    "High-Risk Patients",
    "Patient Timeline",
    "Global Risk Drivers",
    "Research Export",
]

CKD_TARGET = "CKD"
REMISSION_TARGET = "CR_12_MTH_PRED_10"

FEATURE_LABELS = {
    "CREAT BASELINE": "Baseline creatinine",
    "ALBUMIN BASELINE": "Baseline albumin",
    "C4 PRETX": "Pretreatment C4",
    "C3 PRE TX": "Pretreatment C3",
    "CHRONIC INDEX": "Chronicity index",
    "ACTIVE INDEX": "Activity index",
    "GLOBAL SCLEROSIS": "Global sclerosis",
    "MONTH INTERVAL TO INDUCTION TX": "Interval to induction treatment",
    "UPCI PRE TX": "Pretreatment UPCI",
    "LA": "Lupus anticoagulant",
    "ACE/ARB": "ACE/ARB exposure marker",
    "CKD": "CKD marker",
    "RACE": "Race code",
    "GENDER": "Gender code",
}

CLINICAL_MEANINGS = {
    "CREAT BASELINE": "higher baseline creatinine may reflect greater renal impairment in the model",
    "ALBUMIN BASELINE": "albumin level contributes to the model's estimate of clinical status",
    "C4 PRETX": "pretreatment complement C4 contributes to immune activity context in the model",
    "CHRONIC INDEX": "higher chronicity index reflects more chronic biopsy change in the model",
    "ACTIVE INDEX": "activity index contributes biopsy activity context in the model",
    "GLOBAL SCLEROSIS": "global sclerosis contributes biopsy chronic damage context in the model",
    "MONTH INTERVAL TO INDUCTION TX": "longer interval to induction treatment contributes timing context in the model",
    "UPCI PRE TX": "pretreatment proteinuria contributes kidney involvement context in the model",
    "LA": "positive lupus anticoagulant is associated with higher predicted risk in the model",
    "ACE/ARB": "ACE/ARB marker is included as a treatment-context variable in the model",
    "CKD": "CKD status contributes to delayed-remission risk in the model",
    "RACE": "race code is associated with different predicted risk patterns in the model",
}

PREDICTION_CACHE_MAX_SIZE = 256
_PREDICTION_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _risk_category(probability: float) -> str:
    if probability >= 0.67:
        return "High"
    if probability >= 0.34:
        return "Moderate"
    return "Low"


@lru_cache(maxsize=1)
def _raw_df() -> pd.DataFrame:
    return pd.read_excel(DATA_DIR / "ANONYMOUS DATA EXCEL LN RELAPSE_UKM.xlsx", sheet_name="Sheet1")


@lru_cache(maxsize=1)
def _ckd_features() -> list[str]:
    return json.loads((MODEL_DIR / "ckd_rfe_features.json").read_text())


@lru_cache(maxsize=1)
def _remission_features() -> list[str]:
    return json.loads((MODEL_DIR / "remission_rfe_features.json").read_text())


@lru_cache(maxsize=1)
def _ckd_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "CB_CKD_cleaned_with_target.csv")


@lru_cache(maxsize=1)
def _remission_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "CB_Remission_cleaned_with_target.csv")


@lru_cache(maxsize=1)
def _ckd_model() -> Any:
    with (MODEL_DIR / "catboost_ckd_model.pkl").open("rb") as handle:
        return pickle.load(handle)


@lru_cache(maxsize=1)
def _remission_model() -> Any:
    with (MODEL_DIR / "catboost_remission_model.pkl").open("rb") as handle:
        return pickle.load(handle)


@lru_cache(maxsize=2)
def _explainer(model_name: str) -> Any:
    model = _ckd_model() if model_name == "ckd" else _remission_model()
    return shap.TreeExplainer(model)


def _feature_frame(values: dict[str, Any], features: list[str], source: pd.DataFrame) -> pd.DataFrame:
    medians = source[features].median(numeric_only=True).to_dict()
    row = {}
    for feature in features:
        value = values.get(feature)
        if value is None or value == "":
            value = medians.get(feature, 0)
        row[feature] = float(value)
    return pd.DataFrame([row], columns=features)


def _apply_categorical_casts(model: Any, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    for index in model.get_cat_feature_indices():
        feature = features[index]
        frame[feature] = frame[feature].round().astype(int)
    return frame


def _prepared_frame(
    model_name: str, features: list[str], source: pd.DataFrame, values: dict[str, Any]
) -> tuple[Any, pd.DataFrame]:
    frame = _feature_frame(values, features, source)
    model = _ckd_model() if model_name == "ckd" else _remission_model()
    return model, _apply_categorical_casts(model, frame, features)


def _prediction_cache_key(inputs: dict[str, Any], patient_ref: str | None) -> str:
    ckd_features = _ckd_features()
    remission_features = _remission_features()
    _, ckd_frame = _prepared_frame("ckd", ckd_features, _ckd_df(), inputs)
    _, remission_frame = _prepared_frame("remission", remission_features, _remission_df(), inputs)
    key_payload = {
        "patientRef": patient_ref or "Manual input",
        "ckd": [_clean_value(ckd_frame.iloc[0][feature]) for feature in ckd_features],
        "remission": [_clean_value(remission_frame.iloc[0][feature]) for feature in remission_features],
    }
    return json.dumps(key_payload, sort_keys=True, separators=(",", ":"))


def _prediction_cache_get(cache_key: str) -> dict[str, Any] | None:
    cached = _PREDICTION_CACHE.get(cache_key)
    if cached is None:
        return None
    _PREDICTION_CACHE.move_to_end(cache_key)
    return deepcopy(cached)


def _prediction_cache_set(cache_key: str, payload: dict[str, Any]) -> None:
    _PREDICTION_CACHE[cache_key] = deepcopy(payload)
    _PREDICTION_CACHE.move_to_end(cache_key)
    while len(_PREDICTION_CACHE) > PREDICTION_CACHE_MAX_SIZE:
        _PREDICTION_CACHE.popitem(last=False)


def _clear_prediction_cache() -> None:
    _PREDICTION_CACHE.clear()


def _driver_payload(feature: str, value: Any, shap_value: float, rank: int) -> dict[str, Any]:
    return {
        "feature": feature,
        "displayName": FEATURE_LABELS.get(feature, feature.title()),
        "value": _clean_value(value),
        "shapValue": round(float(shap_value), 4),
        "direction": "increases risk" if shap_value >= 0 else "reduces risk",
        "rank": rank,
        "clinicalMeaning": CLINICAL_MEANINGS.get(
            feature, "this feature contributes to the model association for this prediction"
        ),
        "evidenceScope": "model association, not causation",
    }


def _local_shap(model_name: str, frame: pd.DataFrame, features: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values = _explainer(model_name).shap_values(frame)
    if isinstance(values, list):
        values = values[-1]
    shap_row = np.asarray(values)[0]
    ordered = sorted(zip(features, frame.iloc[0].tolist(), shap_row), key=lambda item: abs(item[2]), reverse=True)
    risk = [_driver_payload(feature, value, shap_value, i + 1) for i, (feature, value, shap_value) in enumerate(ordered) if shap_value > 0][:5]
    protective = [
        _driver_payload(feature, value, shap_value, i + 1)
        for i, (feature, value, shap_value) in enumerate(ordered)
        if shap_value < 0
    ][:3]
    return risk, protective


def _predict_one(
    target: str,
    model_name: str,
    model_file: str,
    features: list[str],
    source: pd.DataFrame,
    values: dict[str, Any],
) -> dict[str, Any]:
    model, frame = _prepared_frame(model_name, features, source, values)
    probability = float(model.predict_proba(frame)[0][1])
    top_risk, protective = _local_shap(model_name, frame, features)
    category = _risk_category(probability)
    return {
        "target": target,
        "targetEvent": "Predicted CKD risk" if target == "CKD" else "Predicted delayed-remission risk",
        "modelFile": model_file,
        "probability": round(probability, 4),
        "riskCategory": category,
        "thresholdNotes": "Prototype thresholds: Low < 0.34, Moderate 0.34-0.66, High >= 0.67.",
        "topRiskDrivers": top_risk,
        "protectiveFactors": protective,
    }


def _prediction_probability(model_name: str, features: list[str], source: pd.DataFrame, values: dict[str, Any]) -> float:
    model, frame = _prepared_frame(model_name, features, source, values)
    return float(model.predict_proba(frame)[0][1])


def _summary_drivers(values: dict[str, Any], preferred_features: list[str]) -> list[dict[str, Any]]:
    drivers = []
    for feature in preferred_features:
        value = values.get(feature)
        if value not in (None, "", 0, 0.0):
            drivers.append(
                {
                    "feature": feature,
                    "displayName": FEATURE_LABELS.get(feature, feature.title()),
                    "value": _clean_value(value),
                    "shapValue": 0,
                    "direction": "increases risk",
                    "rank": len(drivers) + 1,
                    "clinicalMeaning": CLINICAL_MEANINGS.get(
                        feature, "this feature contributes to model association for prototype prioritization"
                    ),
                    "evidenceScope": "prototype driver summary; local SHAP is generated in Patient Risk Assessment",
                }
            )
        if len(drivers) == 4:
            break
    return drivers


def _fast_prediction_summary(inputs: dict[str, Any], patient_ref: str) -> dict[str, Any]:
    ckd_probability = _prediction_probability("ckd", _ckd_features(), _ckd_df(), inputs)
    remission_probability = _prediction_probability("remission", _remission_features(), _remission_df(), inputs)
    result = {
        "schemaVersion": "1.0",
        "patientRef": patient_ref,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "predictionKind": "summary",
        "predictionSource": "catboost",
        "shapSource": "mock_shap",
        "inputValidation": {"missingRequiredFields": [], "warnings": []},
        "outcomes": [
            {
                "target": "CKD",
                "targetEvent": "Predicted CKD risk",
                "modelFile": "model/catboost_ckd_model.pkl",
                "probability": round(ckd_probability, 4),
                "riskCategory": _risk_category(ckd_probability),
                "thresholdNotes": "Prototype thresholds: Low < 0.34, Moderate 0.34-0.66, High >= 0.67.",
                "topRiskDrivers": _summary_drivers(
                    inputs, ["CREAT BASELINE", "LA", "CHRONIC INDEX", "C4 PRETX", "RACE", "ALBUMIN BASELINE"]
                ),
                "protectiveFactors": [],
            },
            {
                "target": "Delayed remission",
                "targetEvent": "Predicted delayed-remission risk",
                "modelFile": "model/catboost_remission_model.pkl",
                "probability": round(remission_probability, 4),
                "riskCategory": _risk_category(remission_probability),
                "thresholdNotes": "Prototype thresholds: Low < 0.34, Moderate 0.34-0.66, High >= 0.67.",
                "topRiskDrivers": _summary_drivers(
                    inputs,
                    ["CKD", "MONTH INTERVAL TO INDUCTION TX", "GLOBAL SCLEROSIS", "RACE", "ACE/ARB", "LA"],
                ),
                "protectiveFactors": [],
            },
        ],
    }
    result["llmExplanation"] = explain_payload(result)
    return result


def explain_payload(result: dict[str, Any]) -> dict[str, Any]:
    statements = []
    drivers = []
    protective = []
    for outcome in result["outcomes"]:
        top = outcome["topRiskDrivers"][:3]
        top_names = ", ".join(driver["displayName"] for driver in top) or "available model features"
        statements.append(
            f"The CatBoost {outcome['target']} model predicts {outcome['riskCategory'].lower()} risk "
            f"({outcome['probability']:.0%}), with main contributors including {top_names}."
        )
        drivers.extend([driver["displayName"] for driver in top])
        protective.extend([driver["displayName"] for driver in outcome["protectiveFactors"][:2]])
    return {
        "summary": " ".join(statements),
        "mainDrivers": list(dict.fromkeys(drivers))[:6],
        "protectiveFactors": list(dict.fromkeys(protective))[:4],
        "safetyNote": "Decision-support prototype only; not a diagnostic replacement and not treatment guidance.",
    }


def predict_patient(inputs: dict[str, Any], patient_ref: str | None = None) -> dict[str, Any]:
    cache_key = _prediction_cache_key(inputs, patient_ref)
    cached_result = _prediction_cache_get(cache_key)
    if cached_result is not None:
        return cached_result

    missing = [
        feature
        for feature in sorted(set(_ckd_features() + _remission_features()))
        if inputs.get(feature) in (None, "")
    ]
    result = {
        "schemaVersion": "1.0",
        "patientRef": patient_ref or "Manual input",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "predictionKind": "full",
        "predictionSource": "catboost",
        "shapSource": "local_shap",
        "inputValidation": {
            "missingRequiredFields": missing,
            "warnings": [
                "Missing fields were imputed with model-ready cohort medians for prototype prediction."
            ]
            if missing
            else [],
        },
        "outcomes": [
            _predict_one(
                "CKD",
                "ckd",
                "model/catboost_ckd_model.pkl",
                _ckd_features(),
                _ckd_df(),
                inputs,
            ),
            _predict_one(
                "Delayed remission",
                "remission",
                "model/catboost_remission_model.pkl",
                _remission_features(),
                _remission_df(),
                inputs,
            ),
        ],
    }
    result["llmExplanation"] = explain_payload(result)
    _prediction_cache_set(cache_key, result)
    return deepcopy(result)


def _distribution(series: pd.Series) -> list[dict[str, Any]]:
    counts = series.dropna().value_counts().sort_index()
    return [{"name": str(_clean_value(index)), "value": int(value)} for index, value in counts.items()]


def _summary_stats(raw: pd.DataFrame) -> list[dict[str, Any]]:
    fields = [
        "CREAT BASELINE",
        "CREAT 24 MONTH",
        "ALBUMIN BASELINE",
        "C3 PRE TX",
        "C4 PRETX",
        "UPCI PRE TX",
        "CHRONIC INDEX",
        "ACTIVE INDEX",
    ]
    stats = []
    for field in fields:
        if field in raw.columns:
            values = pd.to_numeric(raw[field], errors="coerce").dropna()
            if not values.empty:
                stats.append(
                    {
                        "field": field,
                        "label": FEATURE_LABELS.get(field, field.title()),
                        "median": round(float(values.median()), 2),
                        "mean": round(float(values.mean()), 2),
                        "available": int(values.count()),
                    }
                )
    return stats


def _timeline(row: pd.Series) -> list[dict[str, Any]]:
    creat = _clean_value(row.get("CREAT BASELINE"))
    creat24 = _clean_value(row.get("CREAT 24 MONTH"))
    points = [
        {"month": 0, "label": "Baseline", "creatinine": creat, "upci": _clean_value(row.get("UPCI PRE TX"))},
        {"month": 3, "label": "3M", "creatinine": None, "upci": _clean_value(row.get("UPCI 3 MTH"))},
        {"month": 6, "label": "6M", "creatinine": None, "upci": _clean_value(row.get("UPCI 6 MTH"))},
        {"month": 12, "label": "12M", "creatinine": None, "upci": _clean_value(row.get("UPCI 12 MTH"))},
        {"month": 18, "label": "18M", "creatinine": None, "upci": _clean_value(row.get("UPCI 18 MTH"))},
        {"month": 24, "label": "24M", "creatinine": creat24, "upci": _clean_value(row.get("UPCI 24MTH"))},
    ]
    return points


@lru_cache(maxsize=1)
def _patient_rows() -> list[dict[str, Any]]:
    raw = _raw_df()
    ckd = _ckd_df()
    remission = _remission_df()
    rows = []
    limit = min(len(ckd), len(remission), len(raw))
    for index in range(limit):
        raw_row = raw.iloc[index]
        inputs = {}
        for feature in sorted(set(_ckd_features() + _remission_features())):
            if feature in ckd.columns:
                inputs[feature] = _clean_value(ckd.iloc[index][feature])
            elif feature in remission.columns:
                inputs[feature] = _clean_value(remission.iloc[index][feature])
            elif feature in raw.columns:
                inputs[feature] = _clean_value(raw_row[feature])
        ref = str(_clean_value(raw_row.get("PATIENT ID")) or f"P{index + 1:03d}")
        prediction = predict_patient(inputs, ref)
        ckd_outcome, remission_outcome = prediction["outcomes"]
        main_reason = ", ".join(driver["displayName"] for driver in ckd_outcome["topRiskDrivers"][:2])
        if remission_outcome["riskCategory"] == "High":
            main_reason = ", ".join(driver["displayName"] for driver in remission_outcome["topRiskDrivers"][:2])
        combined = round((ckd_outcome["probability"] + remission_outcome["probability"]) / 2, 4)
        rows.append(
            {
                "id": ref,
                "displayId": f"Sample {index + 1} / Patient {ref}",
                "rawIndex": index,
                "race": _clean_value(raw_row.get("RACE")),
                "gender": _clean_value(raw_row.get("GENDER")),
                "ageAtInduction": _clean_value(raw_row.get("AGE AT INDUCTION TX")),
                "biopsyDate": _clean_value(raw_row.get("DATE OF BIOPSY")),
                "yearActiveLn": _clean_value(raw_row.get("YEAR ACTIVE LN")),
                "inductionIntervalMonths": _clean_value(raw_row.get("MONTH INTERVAL TO INDUCTION TX")),
                "latestMarker": _clean_value(raw_row.get("CREAT 24 MONTH")),
                "inputs": inputs,
                "timeline": _timeline(raw_row),
                "prediction": prediction,
                "priority": {
                    "combinedRisk": combined,
                    "urgency": _risk_category(combined),
                    "mainReason": main_reason or "Model drivers available in assessment view",
                },
            }
        )
    return rows


@lru_cache(maxsize=1)
def get_dashboard_payload() -> dict[str, Any]:
    raw = _raw_df()
    ckd = _ckd_df()
    remission = _remission_df()
    return {
        "modules": MODULES,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "rawRecords": int(len(raw)),
            "ckdModelReadyRecords": int(len(ckd)),
            "remissionModelReadyRecords": int(len(remission)),
            "ckdDistribution": _distribution(ckd[CKD_TARGET]),
            "delayedRemissionDistribution": _distribution(remission[REMISSION_TARGET]),
            "raceDistribution": _distribution(raw["RACE"]),
            "genderDistribution": _distribution(raw["GENDER"]),
            "summaryStats": _summary_stats(raw),
        },
        "patients": _patient_rows(),
        "featureLists": {"ckd": _ckd_features(), "remission": _remission_features()},
        "shapAssets": {
            "ckd": {
                "barPlot": "/api/assets/ckd_barplot.png",
                "dependencePlots": [
                    {"title": "Baseline creatinine", "src": "/api/assets/Dependence Plots/CKD_CREAT.png"},
                    {"title": "Lupus anticoagulant", "src": "/api/assets/Dependence Plots/CKD_LA.png"},
                    {"title": "Race code", "src": "/api/assets/Dependence Plots/CKD_RACE.png"},
                ],
            },
            "remission": {
                "barPlot": "/api/assets/remission_barplot.png",
                "dependencePlots": [
                    {"title": "CKD marker", "src": "/api/assets/Dependence Plots/Remission_CKD.png"},
                    {"title": "Global sclerosis", "src": "/api/assets/Dependence Plots/Remission_GLOBAL.png"},
                    {"title": "Induction interval", "src": "/api/assets/Dependence Plots/Remission_MONTH.png"},
                    {"title": "Race code", "src": "/api/assets/Dependence Plots/Remission_RACE.png"},
                ],
            },
        },
    }
