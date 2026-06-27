# Patient Digital-Twin Constellation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Patient Twins" dashboard module that finds a patient's nearest real historical twins in the UKM Lupus Nephritis cohort, renders them as an interactive 2D star-map colored by the twins' real outcomes, and reports the twin outcome breakdown with real proteinuria trajectories.

**Architecture:** A new pure service function `find_twins()` in `backend/services.py` (standardize cohort over the model feature union, fit a cached 2D PCA, rank by standardized Euclidean distance, attach real recorded outcomes/trajectories) exposed via `POST /api/twins`. A new self-contained React component `src/components/TwinsModule.tsx` renders a recharts ScatterChart constellation plus an outcome/twins side panel, registered as the 7th module.

**Tech Stack:** Python 3.13, Flask, pandas, numpy, scikit-learn (already installed via SHAP); React 19 + TypeScript, recharts, lucide-react, Tailwind. **No new dependencies.**

## Global Constraints

- No new runtime dependencies. Use only: numpy, pandas, scikit-learn (`sklearn.decomposition.PCA`) on backend; recharts, lucide-react, Tailwind on frontend.
- Determinism: `PCA(n_components=2, random_state=0)` — twin results must be reproducible across runs.
- Safety framing (verbatim copy required): `"Case-based model association, not diagnosis or treatment guidance; requires clinical validation before real-world use."`
- Outcome polarity (recorded data, not model class): `ckd` = `CKD == 1`; `delayedRemission` = `CR_12_MTH_PRED_10 == 0` (complete remission NOT achieved by 12 months); `relapse` = `RELAPSE WITHIN 12 MONTH == 1`. Any other / missing value → `None` (unknown).
- Dataframe alignment: follow the existing positional convention in `_patient_rows()` — row `i` of `_patient_rows()` corresponds to `_ckd_df().iloc[i]`, `_remission_df().iloc[i]`, and `_raw_df().iloc[i]`. Do not introduce a new join.
- Feature-union order (verbatim): `sorted(set(_ckd_features() + _remission_features()))` — identical to what `_patient_rows()` uses to build `inputs`.
- Twin count `n`: default 12, clamped to `[3, 30]`.
- Tests run from the `SLECare-Dashboard/` directory with `python -m pytest`. The repo is **not** git-initialized; "Checkpoint" steps replace commits (committing is optional and only if the user later runs `git init`).
- Frontend has no unit-test runner; the frontend verification gate is `npm run build` (runs `tsc`) plus a manual smoke check.

---

### Task 1: Backend twin matrix, stats, and PCA helpers

**Files:**
- Modify: `backend/services.py` (add constants + helpers near the existing `_modifiable_ranges` block, after line ~503)
- Create: `tests/test_twins.py`

**Interfaces:**
- Consumes: existing `_ckd_features()`, `_remission_features()`, `_patient_rows()`, `_clean_value()`.
- Produces:
  - `_twin_feature_order() -> list[str]`
  - `_twin_float(value: Any) -> float` (returns `float('nan')` for missing/non-numeric)
  - `_twin_matrix_imputed() -> tuple[np.ndarray, list[str]]` (shape `[n_patients, n_features]`, median-imputed, **not** standardized; cached)
  - `_twin_median() -> np.ndarray` (per-feature median of the imputed matrix; cached)
  - `_twin_stats() -> tuple[np.ndarray, np.ndarray]` (per-feature mean, std with std==0 → 1; cached)
  - `_twin_standardized() -> np.ndarray` (z-scored imputed matrix; cached)
  - `_twin_pca() -> tuple[Any, np.ndarray]` (fitted `PCA`, 2D coords for all cohort rows; cached)

- [ ] **Step 1: Write the failing test**

Create `tests/test_twins.py`:

```python
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
    assert features == _twin_feature_order()
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_twins.py -v`
Expected: FAIL with `ImportError: cannot import name '_twin_feature_order'`

- [ ] **Step 3: Write minimal implementation**

In `backend/services.py`, add `import numpy as np` is already present (line 14). Add after the `_modifiable_ranges()` function (around line 503):

```python
TWIN_DEFAULT_N = 12
TWIN_MIN_N = 3
TWIN_MAX_N = 30

# (month, raw column) pairs for a twin's real proteinuria trajectory.
TWIN_TRAJECTORY_COLS = [
    (0, "UPCI PRE TX"),
    (3, "UPCI 3 MTH"),
    (6, "UPCI 6 MTH"),
    (12, "UPCI 12 MTH"),
    (18, "UPCI 18 MTH"),
    (24, "UPCI 24MTH"),
]


def _twin_feature_order() -> list[str]:
    """Feature union used for twin similarity - matches _patient_rows() inputs."""
    return sorted(set(_ckd_features() + _remission_features()))


def _twin_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed


@lru_cache(maxsize=1)
def _twin_matrix_imputed() -> tuple[np.ndarray, tuple[str, ...]]:
    features = _twin_feature_order()
    rows = _patient_rows()
    mat = np.array(
        [[_twin_float(row["inputs"].get(feature)) for feature in features] for row in rows],
        dtype=float,
    )
    col_median = np.nanmedian(mat, axis=0)
    col_median = np.where(np.isnan(col_median), 0.0, col_median)
    nan_mask = np.isnan(mat)
    mat[nan_mask] = np.take(col_median, np.where(nan_mask)[1])
    return mat, tuple(features)


@lru_cache(maxsize=1)
def _twin_median() -> np.ndarray:
    mat, _ = _twin_matrix_imputed()
    return np.median(mat, axis=0)


@lru_cache(maxsize=1)
def _twin_stats() -> tuple[np.ndarray, np.ndarray]:
    mat, _ = _twin_matrix_imputed()
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


@lru_cache(maxsize=1)
def _twin_standardized() -> np.ndarray:
    mat, _ = _twin_matrix_imputed()
    mean, std = _twin_stats()
    return (mat - mean) / std


@lru_cache(maxsize=1)
def _twin_pca() -> tuple[Any, np.ndarray]:
    from sklearn.decomposition import PCA

    z = _twin_standardized()
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(z)
    return pca, coords
```

Note: `_twin_feature_order()` returns a list; `_twin_matrix_imputed()` returns the features as a tuple (so the cached return value is immutable/hashable-friendly). The test compares `features == _twin_feature_order()` — update the test to compare against the tuple. Fix the test's first assertion to: `assert list(features) == _twin_feature_order()`.

- [ ] **Step 4: Run test to verify it passes**

First apply the test fix from Step 3:

```python
    assert list(features) == _twin_feature_order()
```

Run: `python -m pytest tests/test_twins.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Checkpoint**

Confirm all three tests pass. (Optional commit if git is initialized: `git add backend/services.py tests/test_twins.py`.)

---

### Task 2: Backend outcome/trajectory extraction + `find_twins()`

**Files:**
- Modify: `backend/services.py` (add after the Task 1 helpers)
- Modify: `tests/test_twins.py` (append tests)

**Interfaces:**
- Consumes (from Task 1): `_twin_feature_order`, `_twin_float`, `_twin_median`, `_twin_stats`, `_twin_standardized`, `_twin_pca`, `TWIN_DEFAULT_N/MIN_N/MAX_N`, `TWIN_TRAJECTORY_COLS`. Also existing `_ckd_df`, `_remission_df`, `_raw_df`, `_patient_rows`, `find_patient_inputs`, `_clean_value`, `FEATURE_LABELS`, `REMISSION_TARGET`, `SAFETY_NOTE`.
- Produces:
  - `_twin_outcomes(index: int) -> dict[str, bool | None]` with keys `ckd`, `delayedRemission`, `relapse`.
  - `_twin_trajectory(index: int) -> list[dict]` with `month`, `upci`, `creatinine`.
  - `find_twins(patient_ref: str | None, inputs: dict[str, Any] | None, n: int = TWIN_DEFAULT_N) -> dict[str, Any]` returning the full payload (see spec section 4.3).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_twins.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_twins.py -k find_twins -v`
Expected: FAIL with `ImportError: cannot import name 'find_twins'`

- [ ] **Step 3: Write minimal implementation**

In `backend/services.py`, add after the Task 1 helpers:

```python
TWIN_SAFETY_NOTE = (
    "Case-based model association, not diagnosis or treatment guidance; "
    "requires clinical validation before real-world use."
)


def _twin_bool(value: Any, positive: float) -> bool | None:
    cleaned = _clean_value(value)
    if cleaned is None:
        return None
    try:
        return float(cleaned) == positive
    except (TypeError, ValueError):
        return None


def _twin_outcomes(index: int) -> dict[str, bool | None]:
    ckd_val = _ckd_df().iloc[index].get(CKD_TARGET)
    rem_val = _remission_df().iloc[index].get(REMISSION_TARGET)
    relapse_val = _raw_df().iloc[index].get("RELAPSE WITHIN 12 MONTH")
    return {
        "ckd": _twin_bool(ckd_val, 1.0),
        # delayed remission = complete remission NOT achieved by 12 months
        "delayedRemission": _twin_bool(rem_val, 0.0),
        "relapse": _twin_bool(relapse_val, 1.0),
    }


def _twin_trajectory(index: int) -> list[dict[str, Any]]:
    raw_row = _raw_df().iloc[index]
    points = []
    for month, column in TWIN_TRAJECTORY_COLS:
        if month == 0:
            creatinine = _clean_value(raw_row.get("CREAT BASELINE"))
        elif month == 24:
            creatinine = _clean_value(raw_row.get("CREAT 24 MONTH"))
        else:
            creatinine = None
        points.append(
            {"month": month, "upci": _clean_value(raw_row.get(column)), "creatinine": creatinine}
        )
    return points


def _twin_matched_on(
    z_query: np.ndarray, z_twin: np.ndarray, twin_inputs: dict[str, Any], query_inputs: dict[str, Any]
) -> list[dict[str, Any]]:
    features = _twin_feature_order()
    diffs = np.abs(z_query - z_twin)
    order = np.argsort(diffs)[:3]
    matched = []
    for j in order:
        feature = features[int(j)]
        matched.append(
            {
                "feature": feature,
                "displayName": FEATURE_LABELS.get(feature, feature.title()),
                "queryValue": _clean_value(query_inputs.get(feature)),
                "twinValue": _clean_value(twin_inputs.get(feature)),
            }
        )
    return matched


def _project_query(inputs: dict[str, Any]) -> tuple[np.ndarray, list[float]]:
    features = _twin_feature_order()
    mean, std = _twin_stats()
    median = _twin_median()
    raw = np.array([_twin_float(inputs.get(feature)) for feature in features], dtype=float)
    raw = np.where(np.isnan(raw), median, raw)
    z = (raw - mean) / std
    pca, _ = _twin_pca()
    xy = pca.transform(z.reshape(1, -1))[0]
    return z, [float(xy[0]), float(xy[1])]


def find_twins(
    patient_ref: str | None,
    inputs: dict[str, Any] | None,
    n: int = TWIN_DEFAULT_N,
) -> dict[str, Any]:
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = TWIN_DEFAULT_N
    n = max(TWIN_MIN_N, min(TWIN_MAX_N, n))

    if not inputs:
        inputs = find_patient_inputs(patient_ref) or {}

    z_query, query_xy = _project_query(inputs)
    z_cohort = _twin_standardized()
    _, coords = _twin_pca()
    rows = _patient_rows()

    distances = np.linalg.norm(z_cohort - z_query, axis=1)
    d_max = float(distances.max()) or 1.0
    order = np.argsort(distances)
    twin_indices = order[: min(n, len(rows))]

    cohort = [
        {
            "id": rows[i]["id"],
            "x": float(coords[i][0]),
            "y": float(coords[i][1]),
            "outcomes": _twin_outcomes(i),
        }
        for i in range(len(rows))
    ]

    twins = []
    for i in twin_indices:
        i = int(i)
        twins.append(
            {
                "id": rows[i]["id"],
                "displayId": rows[i]["displayId"],
                "x": float(coords[i][0]),
                "y": float(coords[i][1]),
                "similarity": round(100.0 * (1.0 - float(distances[i]) / d_max), 1),
                "outcomes": _twin_outcomes(i),
                "trajectory": _twin_trajectory(i),
                "matchedOn": _twin_matched_on(z_query, z_cohort[i], rows[i]["inputs"], inputs),
            }
        )

    breakdown = {}
    for lens in ("ckd", "delayedRemission", "relapse"):
        known = [t["outcomes"][lens] for t in twins if t["outcomes"][lens] is not None]
        positive = sum(1 for value in known if value)
        breakdown[lens] = {
            "positive": positive,
            "total": len(known),
            "pct": round(100.0 * positive / len(known), 1) if known else 0.0,
        }

    return {
        "schemaVersion": "1.0",
        "patientRef": patient_ref or "Manual input",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "n": n,
        "query": {"x": query_xy[0], "y": query_xy[1]},
        "cohort": cohort,
        "twins": twins,
        "outcomeBreakdown": breakdown,
        "safetyNote": TWIN_SAFETY_NOTE,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_twins.py -v`
Expected: PASS (7 passed total)

- [ ] **Step 5: Verify outcome polarity against real data (one-off sanity check)**

Run:

```bash
python -c "from backend.services import find_twins, _patient_rows as r; t=find_twins(r()[3]['id'], r()[3]['inputs']); print('breakdown', t['outcomeBreakdown']); print('sample twin outcomes', t['twins'][0]['outcomes'])"
```

Expected: `breakdown` prints three dicts with `positive <= total`; `total` may be < 12 where outcomes are unknown. No exceptions.

- [ ] **Step 6: Checkpoint**

Confirm 7 tests pass and the sanity check runs clean.

---

### Task 3: Flask `/api/twins` route

**Files:**
- Modify: `backend/app.py` (import `find_twins`; add route)
- Modify: `tests/test_twins.py` (append a route test using Flask test client)

**Interfaces:**
- Consumes (from Task 2): `find_twins(patient_ref, inputs, n)`.
- Produces: `POST /api/twins` returning `jsonify(find_twins(...))`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_twins.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_twins.py -k api_twins -v`
Expected: FAIL with 404 (route not registered) on the first assertion.

- [ ] **Step 3: Write minimal implementation**

In `backend/app.py`, add `find_twins` to the import block from `.services`:

```python
from .services import (
    ASSET_DIR,
    find_twins,
    get_dashboard_payload,
    predict_patient,
    simulate_patient,
    top_modifiable_drivers,
)
```

Add the route after the `drivers()` route:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_twins.py -v`
Expected: PASS (9 passed total)

- [ ] **Step 5: Checkpoint**

Confirm all 9 tests pass.

---

### Task 4: Frontend types + API client

**Files:**
- Modify: `src/types.ts` (add twin types)
- Modify: `src/services/api.ts` (add `loadTwins`)

**Interfaces:**
- Consumes: existing `PatientRow["inputs"]` shape `Record<string, number | string | null>`.
- Produces:
  - Types `TwinOutcomes`, `TwinTrajectoryPoint`, `TwinMatchedFeature`, `TwinPoint`, `CohortPoint`, `TwinOutcomeBreakdown`, `TwinsResult`.
  - `loadTwins(patientRef: string, inputs: Record<string, number | string | null>, n?: number): Promise<TwinsResult>`.

- [ ] **Step 1: Add the type definitions**

Append to `src/types.ts`:

```typescript
export type TwinLens = "ckd" | "delayedRemission" | "relapse";

export type TwinOutcomes = Record<TwinLens, boolean | null>;

export type TwinTrajectoryPoint = {
  month: number;
  upci: number | null;
  creatinine: number | null;
};

export type TwinMatchedFeature = {
  feature: string;
  displayName: string;
  queryValue: number | string | boolean | null;
  twinValue: number | string | boolean | null;
};

export type CohortPoint = {
  id: string;
  x: number;
  y: number;
  outcomes: TwinOutcomes;
};

export type TwinPoint = CohortPoint & {
  displayId: string;
  similarity: number;
  trajectory: TwinTrajectoryPoint[];
  matchedOn: TwinMatchedFeature[];
};

export type TwinOutcomeBreakdown = Record<
  TwinLens,
  { positive: number; total: number; pct: number }
>;

export type TwinsResult = {
  schemaVersion: string;
  patientRef: string;
  generatedAt: string;
  n: number;
  query: { x: number; y: number };
  cohort: CohortPoint[];
  twins: TwinPoint[];
  outcomeBreakdown: TwinOutcomeBreakdown;
  safetyNote: string;
};
```

Also add `"Patient Twins"` to the `ModuleName` union (top of file):

```typescript
export type ModuleName =
  | "Cohort Overview"
  | "Patient Risk Assessment"
  | "High-Risk Patients"
  | "Patient Timeline"
  | "Global Risk Drivers"
  | "Research Export"
  | "Patient Twins";
```

- [ ] **Step 2: Add the API client function**

In `src/services/api.ts`, extend the import and add the function:

```typescript
import type {
  DashboardPayload,
  DriverEngineResult,
  RiskResult,
  SimulationResult,
  TwinsResult,
} from "../types";
```

```typescript
export async function loadTwins(
  patientRef: string,
  inputs: Record<string, number | string | null>,
  n = 12,
): Promise<TwinsResult> {
  const response = await fetch("/api/twins", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patientRef, inputs, n }),
  });
  if (!response.ok) {
    throw new Error("Twin constellation service is unavailable.");
  }
  return response.json();
}
```

- [ ] **Step 3: Verify type-check passes**

Run: `npm run build`
Expected: build succeeds (tsc reports no errors). The `"Patient Twins"` union member is referenced by the backend MODULES list added in Task 6; at this point it only needs to compile.

- [ ] **Step 4: Checkpoint**

`npm run build` is green.

---

### Task 5: TwinsModule React component

**Files:**
- Create: `src/components/TwinsModule.tsx`
- Modify: `src/styles.css` (add the pulse keyframe + constellation helpers)

**Interfaces:**
- Consumes (from Task 4): `loadTwins`, types `TwinsResult`, `TwinPoint`, `TwinLens`, `PatientRow`.
- Produces: default-less named export `export function TwinsModule({ selectedPatient }: { selectedPatient: PatientRow })`.

- [ ] **Step 1: Add CSS helpers**

Append to `src/styles.css`:

```css
@keyframes twin-pulse {
  0% { r: 7; opacity: 1; }
  50% { r: 11; opacity: 0.55; }
  100% { r: 7; opacity: 1; }
}
.twin-you-marker { animation: twin-pulse 1.8s ease-in-out infinite; }
.twin-lens-btn { padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.8rem; }
.twin-lens-btn.active { background: #0f766e; color: #fff; }
```

- [ ] **Step 2: Write the component**

Create `src/components/TwinsModule.tsx`:

```typescript
import { useEffect, useMemo, useState } from "react";
import { Loader2, Orbit, Sparkles } from "lucide-react";
import {
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import { loadTwins } from "../services/api";
import type { PatientRow, TwinLens, TwinPoint, TwinsResult } from "../types";

const LENS_LABEL: Record<TwinLens, string> = {
  ckd: "Developed CKD",
  delayedRemission: "Delayed remission",
  relapse: "Relapsed (12 mo)",
};

// adverse=true -> rose, false (favourable) -> teal, unknown -> slate
function lensColor(value: boolean | null): string {
  if (value === true) return "#e11d48";
  if (value === false) return "#0d9488";
  return "#94a3b8";
}

export function TwinsModule({ selectedPatient }: { selectedPatient: PatientRow }) {
  const [data, setData] = useState<TwinsResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lens, setLens] = useState<TwinLens>("ckd");
  const [selectedTwinId, setSelectedTwinId] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    loadTwins(selectedPatient.id, selectedPatient.inputs, 12)
      .then((result) => {
        if (!active) return;
        setData(result);
        setSelectedTwinId(result.twins[0]?.id ?? null);
      })
      .catch((err: Error) => active && setError(err.message))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [selectedPatient]);

  const twinIds = useMemo(() => new Set(data?.twins.map((t) => t.id) ?? []), [data]);
  const backgroundCohort = useMemo(
    () => data?.cohort.filter((point) => !twinIds.has(point.id)) ?? [],
    [data, twinIds],
  );
  const selectedTwin: TwinPoint | undefined = useMemo(
    () => data?.twins.find((t) => t.id === selectedTwinId),
    [data, selectedTwinId],
  );

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center text-slate-500">
        <Loader2 className="mr-2 animate-spin" size={18} /> Mapping patient twins...
      </div>
    );
  }
  if (error) return <div className="card text-rose-700">{error}</div>;
  if (!data) return null;

  const breakdown = data.outcomeBreakdown[lens];

  return (
    <section className="space-y-4">
      <header className="flex items-center gap-3">
        <Orbit className="text-clinical" size={22} />
        <div>
          <h2 className="text-xl font-semibold text-ink">Patient Twin Constellation</h2>
          <p className="text-sm text-slate-500">
            {selectedPatient.displayId} &middot; nearest {data.n} real cohort twins by standardized
            clinical similarity.
          </p>
        </div>
      </header>

      <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
        {/* Constellation */}
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-4">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <span className="text-xs uppercase tracking-wide text-slate-400">Outcome lens:</span>
            {(Object.keys(LENS_LABEL) as TwinLens[]).map((key) => (
              <button
                key={key}
                className={`twin-lens-btn ${lens === key ? "active" : "text-slate-300"}`}
                onClick={() => setLens(key)}
              >
                {LENS_LABEL[key]}
              </button>
            ))}
          </div>
          <ResponsiveContainer width="100%" height={420}>
            <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
              <CartesianGrid stroke="#1e293b" />
              <XAxis type="number" dataKey="x" hide domain={["dataMin - 1", "dataMax + 1"]} />
              <YAxis type="number" dataKey="y" hide domain={["dataMin - 1", "dataMax + 1"]} />
              <ZAxis type="number" dataKey="similarity" range={[40, 320]} />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                content={({ payload }) => {
                  const point = payload?.[0]?.payload;
                  if (!point) return null;
                  return (
                    <div className="rounded bg-white p-2 text-xs shadow">
                      <p className="font-semibold">{point.displayId ?? point.id}</p>
                      {point.similarity !== undefined && <p>{point.similarity}% similar</p>}
                      <p style={{ color: lensColor(point.outcomes?.[lens]) }}>
                        {LENS_LABEL[lens]}:{" "}
                        {point.outcomes?.[lens] === null ? "unknown" : point.outcomes?.[lens] ? "yes" : "no"}
                      </p>
                    </div>
                  );
                }}
              />
              {/* background cohort */}
              <Scatter data={backgroundCohort} fill="#334155" />
              {/* twins */}
              <Scatter
                data={data.twins}
                onClick={(point) => setSelectedTwinId((point as TwinPoint).id)}
              >
                {data.twins.map((twin) => (
                  <Cell
                    key={twin.id}
                    fill={lensColor(twin.outcomes[lens])}
                    stroke={twin.id === selectedTwinId ? "#fff" : "none"}
                    strokeWidth={2}
                  />
                ))}
              </Scatter>
              {/* query patient */}
              <Scatter
                data={[{ ...data.query, id: "YOU", displayId: `${selectedPatient.displayId} (this patient)` }]}
                shape={(props: any) => (
                  <circle className="twin-you-marker" cx={props.cx} cy={props.cy} r={8} fill="#facc15" stroke="#fff" strokeWidth={2} />
                )}
              />
            </ScatterChart>
          </ResponsiveContainer>
          <p className="mt-2 flex items-center gap-1 text-xs text-slate-400">
            <Sparkles size={12} /> {data.safetyNote}
          </p>
        </div>

        {/* Side panel */}
        <div className="space-y-4">
          <div className="card">
            <h3 className="text-sm font-semibold text-ink">Outcome of {data.twins.length} twins</h3>
            <p className="mt-1 text-2xl font-bold" style={{ color: lensColor(true) }}>
              {breakdown.positive}/{breakdown.total}
              <span className="ml-1 text-sm font-normal text-slate-500">({breakdown.pct}%)</span>
            </p>
            <p className="text-xs text-slate-500">{LENS_LABEL[lens]} among twins with a known outcome.</p>
          </div>

          <div className="card">
            <h3 className="mb-2 text-sm font-semibold text-ink">Twins by similarity</h3>
            <ul className="max-h-48 space-y-1 overflow-auto text-sm">
              {data.twins.map((twin) => (
                <li key={twin.id}>
                  <button
                    className={`flex w-full items-center justify-between rounded px-2 py-1 ${
                      twin.id === selectedTwinId ? "bg-teal-50" : "hover:bg-slate-50"
                    }`}
                    onClick={() => setSelectedTwinId(twin.id)}
                  >
                    <span>{twin.displayId}</span>
                    <span className="font-medium">{twin.similarity}%</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>

          {selectedTwin && (
            <div className="card">
              <h3 className="text-sm font-semibold text-ink">{selectedTwin.displayId}</h3>
              <p className="mb-1 text-xs text-slate-500">Real UPCI trajectory (g/g)</p>
              <ResponsiveContainer width="100%" height={120}>
                <LineChart data={selectedTwin.trajectory}>
                  <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} width={28} />
                  <Tooltip />
                  <Line type="monotone" dataKey="upci" stroke="#0d9488" dot connectNulls />
                </LineChart>
              </ResponsiveContainer>
              <p className="mt-2 text-xs font-medium text-slate-600">Matched on:</p>
              <ul className="text-xs text-slate-500">
                {selectedTwin.matchedOn.map((m) => (
                  <li key={m.feature}>
                    {m.displayName}: you {String(m.queryValue ?? "NA")} / twin {String(m.twinValue ?? "NA")}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
```

- [ ] **Step 3: Verify type-check passes**

Run: `npm run build`
Expected: build succeeds. (If `lucide-react` lacks the `Orbit` icon in the installed version, substitute `Sparkles` for the header icon and remove the `Orbit` import.)

- [ ] **Step 4: Checkpoint**

`npm run build` is green; `src/components/TwinsModule.tsx` exists.

---

### Task 6: Register the module (backend list + frontend wiring)

**Files:**
- Modify: `backend/services.py:23-30` (append to `MODULES`)
- Modify: `src/App.tsx` (import `TwinsModule`, add nav icon, add render slot)

**Interfaces:**
- Consumes: `TwinsModule` (Task 5), `selectedPatient` (already in `App`).
- Produces: a 7th navigable module wired end-to-end.

- [ ] **Step 1: Add the backend module entry**

In `backend/services.py`, append to the `MODULES` list:

```python
MODULES = [
    "Cohort Overview",
    "Patient Risk Assessment",
    "High-Risk Patients",
    "Patient Timeline",
    "Global Risk Drivers",
    "Research Export",
    "Patient Twins",
]
```

- [ ] **Step 2: Add a regression test for the module list**

Append to `tests/test_twins.py`:

```python
from backend.services import MODULES  # noqa: E402


def test_patient_twins_module_registered():
    assert MODULES[-1] == "Patient Twins"
    assert len(MODULES) == 7
```

Run: `python -m pytest tests/test_twins.py -k module_registered -v`
Expected: PASS.

- [ ] **Step 3: Wire the frontend**

In `src/App.tsx`:

(a) Add the import near the other local imports (after line 38):

```typescript
import { TwinsModule } from "./components/TwinsModule";
```

(b) Add an icon to `navIcons` (line 52). Import `Orbit` in the lucide-react block (line 2-22), then:

```typescript
const navIcons = [Users, Stethoscope, ShieldAlert, LineChartIcon, BarChart3, FileText, Orbit];
```

(c) Add the render slot after the Research Export line (line 1052):

```tsx
        {active === "Research Export" && <ResearchExport data={data} />}
        {active === "Patient Twins" && <TwinsModule selectedPatient={selectedPatient} />}
```

- [ ] **Step 4: Verify full type-check + backend tests**

Run: `npm run build`
Expected: build succeeds.

Run: `python -m pytest tests/test_twins.py -v`
Expected: PASS (11 passed total).

- [ ] **Step 5: Checkpoint**

Both gates green.

---

### Task 7: End-to-end smoke verification

**Files:** none (verification only).

- [ ] **Step 1: Start the backend**

Run (background): `python -m backend.app`
Expected: Flask serving on `http://127.0.0.1:5000`.

- [ ] **Step 2: Hit the new endpoint**

Run:

```bash
curl -s -X POST http://127.0.0.1:5000/api/twins -H "Content-Type: application/json" -d "{}" | python -c "import sys,json; d=json.load(sys.stdin); print('n', d['n'], 'twins', len(d['twins']), 'breakdown', d['outcomeBreakdown'])"
```

Expected: `n 12 twins 12 breakdown {...}` with no error.

- [ ] **Step 3: Confirm dashboard exposes the module**

Run:

```bash
curl -s http://127.0.0.1:5000/api/dashboard | python -c "import sys,json; print(json.load(sys.stdin)['modules'])"
```

Expected: list ends with `'Patient Twins'`.

- [ ] **Step 4: Manual UI smoke (frontend)**

Run (separate terminal): `npm run dev` and open `http://127.0.0.1:5173`.
Verify:
- "Patient Twins" appears in the sidebar (7th item, Orbit icon).
- Selecting it renders the constellation with a glowing yellow "YOU" marker, dim cohort dots, and ~12 colored twin dots.
- The outcome-lens buttons (CKD / Delayed remission / Relapse) recolor the twins.
- Clicking a twin (dot or list item) shows its UPCI trajectory sparkline + "Matched on" list.
- The safety note is visible under the chart.

- [ ] **Step 5: Stop the background backend and checkpoint**

Confirm all checks pass. Feature complete.

---

## Self-Review

**Spec coverage:**
- Spec §3 data grounding → Tasks 1–2 (feature union, PCA, outcomes from `CKD`/`CR_12_MTH_PRED_10`/`RELAPSE WITHIN 12 MONTH`, UPCI/creatinine trajectory, matchedOn). ✓
- Spec §4 backend endpoint + response shape → Tasks 2 (`find_twins`) & 3 (route); response keys match §4.3. ✓
- Spec §4.4 caching → all helpers `@lru_cache`d (Task 1). ✓
- Spec §5 frontend module, layout, constellation, side panel, lens, styling, safety → Tasks 4–6. ✓
- Spec §6 component boundaries → `find_twins` pure service, thin route, isolated `TwinsModule`. ✓
- Spec §7 testing → backend pytest (Tasks 1–3, 6), frontend `tsc` build + manual smoke (Tasks 4–7). ✓
- Spec §2 non-goals respected: PCA only (no UMAP), no persistence/auth/animation library (CSS keyframe only). ✓

**Placeholder scan:** No TBD/TODO; every code step contains complete code; every command has expected output. ✓

**Type consistency:** `TwinsResult`/`TwinPoint`/`TwinLens`/`TwinOutcomes` defined in Task 4 are the exact names consumed in Tasks 5–6; `find_twins(patient_ref, inputs, n)` signature matches between Tasks 2, 3, and the route; `_twin_*` helper names consistent across Tasks 1–2; outcome keys `ckd`/`delayedRemission`/`relapse` consistent backend↔frontend. ✓
