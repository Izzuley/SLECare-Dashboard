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

# Clinically adjustable continuous variables for the What-If Explorer and the
# Top Modifiable Driver Engine. Demographics (RACE/GENDER) and binary serology
# markers are intentionally excluded - they are not clinically "modifiable" and
# including them would weaken the model-association-only safety framing.
# `lowerIsBetter` records the direction in which the cohort distribution is
# considered healthier, used to pick a safe perturbation target percentile.
MODIFIABLE_FEATURES: list[dict[str, Any]] = [
    {"feature": "CREAT BASELINE", "lowerIsBetter": True},
    {"feature": "ALBUMIN BASELINE", "lowerIsBetter": False},
    {"feature": "UPCI PRE TX", "lowerIsBetter": True},
    {"feature": "C4 PRETX", "lowerIsBetter": False},
    {"feature": "CHRONIC INDEX", "lowerIsBetter": True},
    {"feature": "ACTIVE INDEX", "lowerIsBetter": True},
    {"feature": "GLOBAL SCLEROSIS", "lowerIsBetter": True},
    {"feature": "MONTH INTERVAL TO INDUCTION TX", "lowerIsBetter": True},
]


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


def _numeric_series_for_feature(feature: str) -> pd.Series | None:
    for source in (_ckd_df(), _remission_df(), _raw_df()):
        if feature in source.columns:
            values = pd.to_numeric(source[feature], errors="coerce").dropna()
            if not values.empty:
                return values
    return None


@lru_cache(maxsize=1)
def _modifiable_ranges() -> dict[str, dict[str, float]]:
    """Cohort percentile ranges for each modifiable feature.

    Perturbation targets are drawn from the patient cohort distribution so the
    simulated value stays in-distribution and clinically defensible.
    """
    ranges: dict[str, dict[str, float]] = {}
    for spec in MODIFIABLE_FEATURES:
        feature = spec["feature"]
        series = _numeric_series_for_feature(feature)
        if series is None:
            continue
        ranges[feature] = {
            "p25": float(series.quantile(0.25)),
            "p50": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return ranges


def _confidence_label(reduction: float) -> str:
    # Deliberately cautious: cap at "Moderate" so outputs never read as
    # treatment certainty. Framed as a model-association strength, not advice.
    if reduction >= 0.10:
        return "Moderate"
    if reduction >= 0.03:
        return "Low-Moderate"
    return "Low"


def find_patient_inputs(patient_ref: str | None) -> dict[str, Any] | None:
    """Look up the model-ready inputs for a stored sample patient by reference."""
    if not patient_ref:
        return None
    for row in _patient_rows():
        if row["id"] == patient_ref:
            return deepcopy(row["inputs"])
    return None


def _delta_payload(baseline: dict[str, Any], simulated: dict[str, Any]) -> dict[str, Any]:
    base_p = baseline["probability"]
    sim_p = simulated["probability"]
    delta = round(sim_p - base_p, 4)
    if delta < -0.005:
        direction = "reduced"
    elif delta > 0.005:
        direction = "increased"
    else:
        direction = "unchanged"
    return {
        "target": baseline["target"],
        "targetEvent": baseline["targetEvent"],
        "baselineProbability": base_p,
        "simulatedProbability": sim_p,
        "delta": delta,
        "deltaPercentagePoints": round(delta * 100, 1),
        "baselineCategory": baseline["riskCategory"],
        "simulatedCategory": simulated["riskCategory"],
        "direction": direction,
    }


SAFETY_NOTE = (
    "Analytical model-based simulation only. Outputs are model associations, "
    "not treatment recommendations, and require clinical validation before "
    "real-world use."
)


def simulate_patient(
    patient_ref: str | None,
    modified_inputs: dict[str, Any],
    baseline_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Clinical What-If Explorer: re-run prediction with modified variables.

    Returns the baseline prediction, the simulated prediction (each a full
    CatBoost + local SHAP RiskResult), and per-outcome deltas.
    """
    if baseline_inputs is None:
        baseline_inputs = find_patient_inputs(patient_ref)
    if baseline_inputs is None:
        baseline_inputs = {}

    sanitized: dict[str, Any] = {}
    for key, value in (modified_inputs or {}).items():
        if value in (None, ""):
            continue
        try:
            sanitized[key] = float(value)
        except (TypeError, ValueError):
            sanitized[key] = value

    simulated_inputs = {**baseline_inputs, **sanitized}

    baseline = predict_patient(baseline_inputs, patient_ref)
    sim_label = f"{patient_ref} (simulated)" if patient_ref else "Manual input (simulated)"
    simulated = predict_patient(simulated_inputs, sim_label)

    deltas = [
        _delta_payload(b, s)
        for b, s in zip(baseline["outcomes"], simulated["outcomes"])
    ]

    return {
        "schemaVersion": "1.0",
        "patientRef": patient_ref or "Manual input",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "modifiedInputs": {k: _clean_value(v) for k, v in sanitized.items()},
        "baseline": baseline,
        "simulated": simulated,
        "deltas": deltas,
        "safetyNote": SAFETY_NOTE,
    }


def _target_specs() -> list[dict[str, Any]]:
    return [
        {"target": "CKD", "model": "ckd", "features": _ckd_features(), "source": _ckd_df()},
        {
            "target": "Delayed remission",
            "model": "remission",
            "features": _remission_features(),
            "source": _remission_df(),
        },
    ]


def top_modifiable_drivers(
    patient_ref: str | None,
    baseline_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Top Modifiable Driver Engine.

    Perturbs each clinically adjustable variable toward its healthier cohort
    quartile, re-runs the prediction, and ranks variables by the predicted risk
    reduction for each outcome. Reported as model associations, not advice.
    """
    if baseline_inputs is None:
        baseline_inputs = find_patient_inputs(patient_ref)
    if baseline_inputs is None:
        baseline_inputs = {}

    ranges = _modifiable_ranges()
    specs = _target_specs()
    baseline_probs = {
        spec["target"]: round(
            _prediction_probability(spec["model"], spec["features"], spec["source"], baseline_inputs),
            4,
        )
        for spec in specs
    }

    rankings: dict[str, list[dict[str, Any]]] = {spec["target"]: [] for spec in specs}

    for mod in MODIFIABLE_FEATURES:
        feature = mod["feature"]
        lower_is_better = mod["lowerIsBetter"]
        if feature not in ranges:
            continue

        current = baseline_inputs.get(feature)
        try:
            current_value = float(current) if current not in (None, "") else None
        except (TypeError, ValueError):
            current_value = None
        if current_value is None:
            continue

        target_value = ranges[feature]["p25"] if lower_is_better else ranges[feature]["p75"]
        # Only perturb if the cohort target is genuinely healthier than the
        # patient's current value; otherwise there is no improvement to claim.
        if lower_is_better and target_value >= current_value:
            continue
        if not lower_is_better and target_value <= current_value:
            continue

        perturbed_inputs = {**baseline_inputs, feature: target_value}

        for spec in specs:
            target = spec["target"]
            new_prob = round(
                _prediction_probability(spec["model"], spec["features"], spec["source"], perturbed_inputs),
                4,
            )
            reduction = round(baseline_probs[target] - new_prob, 4)
            if reduction <= 0:
                continue
            change = "decrease" if lower_is_better else "increase"
            rankings[target].append(
                {
                    "feature": feature,
                    "displayName": FEATURE_LABELS.get(feature, feature.title()),
                    "baselineValue": _clean_value(current_value),
                    "suggestedValue": round(target_value, 3),
                    "perturbation": f"Model {change} toward cohort {'25th' if lower_is_better else '75th'} percentile",
                    "baselineRisk": baseline_probs[target],
                    "simulatedRisk": new_prob,
                    "riskReduction": reduction,
                    "riskReductionPct": round(reduction * 100, 1),
                    "confidence": _confidence_label(reduction),
                    "clinicalMeaning": CLINICAL_MEANINGS.get(
                        feature, "this feature contributes to the model association for this prediction"
                    ),
                    "evidenceScope": "model association, not causation",
                }
            )

    top_driver: dict[str, Any] = {}
    for target, drivers in rankings.items():
        drivers.sort(key=lambda item: item["riskReduction"], reverse=True)
        for rank, driver in enumerate(drivers, start=1):
            driver["rank"] = rank
        top_driver[target] = drivers[0] if drivers else None

    return {
        "schemaVersion": "1.0",
        "patientRef": patient_ref or "Manual input",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "baselineRisk": baseline_probs,
        "rankings": rankings,
        "topDriver": top_driver,
        "safetyNote": SAFETY_NOTE,
    }


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
