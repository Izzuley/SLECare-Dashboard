import numpy as np

from backend.services import (
    _patient_rows,
    _twin_feature_order,
    _twin_matrix_imputed,
    _twin_pca,
    _twin_standardized,
    _twin_stats,
)


def test_twin_matrix_shape_and_no_nans():
    mat, features = _twin_matrix_imputed()
    assert list(features) == _twin_feature_order()
    assert mat.shape == (len(_patient_rows()), len(features))
    assert not np.isnan(mat).any()  # median-imputed, fully dense


def test_twin_standardized_is_zero_mean_unit_std():
    z = _twin_standardized()
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-6)
    # std is 1 except for zero-variance columns which were forced to std 1
    stds = z.std(axis=0)
    assert np.all((np.isclose(stds, 1.0, atol=1e-6)) | np.isclose(stds, 0.0, atol=1e-6))


def test_twin_pca_coords_are_2d_and_deterministic():
    _, coords_a = _twin_pca()
    assert coords_a.shape == (len(_patient_rows()), 2)
    assert np.isfinite(coords_a).all()
    mean, std = _twin_stats()
    assert std.min() > 0  # zero-variance guard applied


from backend.services import _patient_rows as _rows  # noqa: E402
from backend.services import find_twins  # noqa: E402


def test_find_twins_returns_clamped_count_and_sorted_similarity():
    ref = _rows()[0]["id"]
    inputs = _rows()[0]["inputs"]
    result = find_twins(ref, inputs, n=12)
    assert result["n"] == 12
    assert len(result["twins"]) == 12
    sims = [t["similarity"] for t in result["twins"]]
    assert sims == sorted(sims, reverse=True)
    assert all(0.0 <= s <= 100.0 for s in sims)
    # nearest twin to a real patient is itself -> ~100% similarity
    assert result["twins"][0]["similarity"] >= 99.0


def test_find_twins_clamps_out_of_range_n():
    ref = _rows()[5]["id"]
    inputs = _rows()[5]["inputs"]
    assert find_twins(ref, inputs, n=1)["n"] == 3
    assert find_twins(ref, inputs, n=999)["n"] == 30


def test_find_twins_outcome_breakdown_matches_twin_flags():
    ref = _rows()[3]["id"]
    result = find_twins(ref, _rows()[3]["inputs"], n=12)
    for lens in ("ckd", "delayedRemission", "relapse"):
        positive = sum(1 for t in result["twins"] if t["outcomes"][lens] is True)
        known = sum(1 for t in result["twins"] if t["outcomes"][lens] is not None)
        assert result["outcomeBreakdown"][lens]["positive"] == positive
        assert result["outcomeBreakdown"][lens]["total"] == known


def test_find_twins_query_and_shape():
    result = find_twins(_rows()[0]["id"], _rows()[0]["inputs"], n=8)
    assert set(result["query"]) == {"x", "y"}
    assert len(result["cohort"]) == len(_rows())
    twin = result["twins"][0]
    assert {"id", "x", "y", "similarity", "outcomes", "trajectory", "matchedOn"} <= set(twin)
    assert len(twin["trajectory"]) == 6
    assert all(p["month"] in (0, 3, 6, 12, 18, 24) for p in twin["trajectory"])
    assert 1 <= len(twin["matchedOn"]) <= 3
    assert result["safetyNote"].startswith("Case-based model association")


from backend.app import app as flask_app  # noqa: E402
from backend.services import _patient_rows as _rows3  # noqa: E402


def test_api_twins_route_returns_payload():
    client = flask_app.test_client()
    ref = _rows3()[0]["id"]
    response = client.post("/api/twins", json={"patientRef": ref, "inputs": _rows3()[0]["inputs"], "n": 10})
    assert response.status_code == 200
    body = response.get_json()
    assert body["n"] == 10
    assert len(body["twins"]) == 10
    assert "outcomeBreakdown" in body


def test_api_twins_route_tolerates_empty_body():
    client = flask_app.test_client()
    response = client.post("/api/twins", json={})
    assert response.status_code == 200
    body = response.get_json()
    assert body["n"] == 12  # default


from backend.services import MODULES  # noqa: E402


def test_patient_twins_module_registered():
    assert MODULES[-1] == "Patient Twins"
    assert len(MODULES) == 7
