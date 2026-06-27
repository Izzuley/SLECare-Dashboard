# Patient Digital-Twin Constellation — Design Spec

Date: 2026-06-26
Status: Approved (design), pending implementation
Project: SLECare-Dashboard (Lupus Nephritis decision-support dashboard)

## 1. Summary

A new dashboard module ("Patient Twins") that, for the currently selected
patient, finds that patient's nearest real historical twins in the UKM Lupus
Nephritis cohort and renders the whole cohort as an interactive 2D star-map.
The query patient sits at the center; the N nearest twins are highlighted and
colored by what *actually happened to those real patients* (developed CKD /
delayed remission / relapsed). A headline reports the outcome breakdown of the
twins, and hovering a twin shows that real patient's 24-month proteinuria
(UPCI) trajectory plus the features that made them a twin.

The feature pairs the existing CatBoost black-box risk models with transparent,
case-based reasoning over real outcomes — the way clinicians actually reason
("I have seen patients like this before"). It is presented as case-based model
association, NOT diagnosis or treatment guidance, consistent with the
dashboard's existing safety framing.

## 2. Goals and non-goals

Goals:
- A visually striking, grounded "wow" visualization for judges.
- Genuinely useful to clinicians: real-outcome, case-based context for a patient.
- Zero new runtime dependencies.

Non-goals (YAGNI):
- No UMAP/t-SNE (PCA is deterministic, dependency-free, defensible).
- No persistence, no auth, no audit log.
- No animation library (CSS pulse only).
- No change to the diagnosis/treatment guardrail — this stays decision-support.

## 3. Data grounding

All inputs already exist in the repo. No new data.

- Similarity feature space: union of the two RFE feature lists
  (`model/ckd_rfe_features.json` + `model/remission_rfe_features.json`), the
  same features and median-imputation already used by `_feature_frame`.
- Cohort matrix rows: the model-ready patients from `_patient_rows()` (have
  known outcomes), aligned with the raw Excel for trajectories/outcomes.
- Map coordinates: 2D PCA fit on the standardized cohort matrix.
- Twin outcome labels (real columns):
  - CKD: `CKD`
  - Delayed remission: `CR_12_MTH_PRED_10`
  - Relapse: `RELAPSE WITHIN 12 MONTH` (fallback `RELAPSE WITHIN 24 MONTH`)
- Twin trajectory (hover sparkline): `UPCI PRE TX`, `UPCI 3 MTH`, `UPCI 6 MTH`,
  `UPCI 12 MTH`, `UPCI 18 MTH`, `UPCI 24MTH`; plus `CREAT BASELINE`,
  `CREAT 24 MONTH`.
- "Why these are twins": the features where the twin's z-score is closest to the
  query's z-score (smallest absolute per-feature standardized difference).

## 4. Backend design

New service function `find_twins()` in `backend/services.py` and a new route
`POST /api/twins` in `backend/app.py`.

### 4.1 Request
```
POST /api/twins
{ "patientRef": "P084" | null, "inputs": { <feature>: value, ... }, "n": 12 }
```
- If `inputs` is empty and `patientRef` is provided, baseline inputs are
  resolved via the existing `find_patient_inputs(patient_ref)`.
- `n` defaults to 12, clamped to a sane range (e.g. 3..30).

### 4.2 Computation
1. Build a standardized cohort matrix over the model feature union:
   - Assemble the feature matrix from the cohort patients (reuse
     `_patient_rows()` inputs, which are already model-ready + cleaned).
   - Median-impute missing values per feature (consistent with existing code).
   - Standardize to z-scores using cohort mean/std (guard zero-variance
     columns by treating std 0 as 1). Cache with `@lru_cache`.
2. Fit `sklearn.decomposition.PCA(n_components=2, random_state=0)` on the
   standardized matrix (cached). Project all cohort rows -> `[x, y]`.
3. Standardize the query inputs with the cohort mean/std; project via the fitted
   PCA to get the query `[x, y]`.
4. Compute Euclidean distance in the full standardized feature space (not the 2D
   projection) from the query to every cohort row; rank ascending; take top `n`
   as twins. Similarity% = `round(100 * (1 - d / d_max), 1)` where `d_max` is
   the max distance among the cohort (so the farthest patient ~0%, identical ~100%).
5. For each twin, attach: cohort id/displayId, 2D coords, similarity%, the three
   real outcomes (CKD / delayed remission / relapse as booleans + labels), the
   real UPCI trajectory + creatinine points, and the top-3 matching features
   (display name + query value + twin value).
6. Outcome breakdown across the N twins: count and % positive for each of the
   three outcome lenses.

### 4.3 Response (shape)
```json
{
  "schemaVersion": "1.0",
  "patientRef": "P084",
  "generatedAt": "<iso>",
  "n": 12,
  "query": { "x": 0.0, "y": 0.0 },
  "cohort": [ { "id": "P001", "x": 1.2, "y": -0.4,
                "outcomes": { "ckd": true, "delayedRemission": false, "relapse": false } } ],
  "twins": [ {
     "id": "P084", "displayId": "Sample 84 / Patient P084",
     "x": 0.1, "y": 0.0, "similarity": 94.2,
     "outcomes": { "ckd": true, "delayedRemission": true, "relapse": false },
     "trajectory": [ { "month": 0, "upci": 3.1, "creatinine": 120 }, ... ],
     "matchedOn": [ { "feature": "CREAT BASELINE", "displayName": "Baseline creatinine",
                      "queryValue": 110, "twinValue": 115 } ]
  } ],
  "outcomeBreakdown": {
     "ckd": { "positive": 8, "total": 12, "pct": 66.7 },
     "delayedRemission": { "positive": 9, "total": 12, "pct": 75.0 },
     "relapse": { "positive": 5, "total": 12, "pct": 41.7 }
  },
  "safetyNote": "Case-based model association, not diagnosis or treatment guidance..."
}
```

### 4.4 Caching / performance
- Standardized matrix, cohort mean/std, and fitted PCA are each `@lru_cache`d
  (process-lifetime), so a request is just one projection + a vectorized
  distance computation over ~174 rows — effectively instant.

## 5. Frontend design

### 5.1 Module registration
- Add `"Patient Twins"` to `MODULES` (backend) and to the module union in
  `src/types.ts`. It renders in the sidebar/mobile select like the other modules.
- Add `loadTwins(patientRef, inputs, n?)` to `src/services/api.ts` ->
  `POST /api/twins`. Add the response types to `src/types.ts`.

### 5.2 Layout (two-pane)
```
+-----------------------------------------------+-----------------------+
|  PATIENT TWIN CONSTELLATION                    |  OUTCOME OF 12 TWINS  |
|        .  .    .  . .    .   . (dim cohort)    |  * CKD          8/12  |
|      .   *        @ <-YOU      .               |  * Delayed rem. 9/12  |
|         *   * *        *  .    .               |  * Relapsed     5/12  |
|    .       *   * (glowing twins, by outcome)   |                       |
|  Lens: [CKD] [Delayed remission] [Relapse]     |  TWIN LIST            |
|  (i) Case-based association, not diagnosis.     |  P084  94% similar -> |
+-----------------------------------------------+-----------------------+
```

### 5.3 Constellation (recharts ScatterChart)
- Layer 1: all cohort points as dim/small dots (background "stars").
- Layer 2: twin points, colored by the active outcome lens, radius scaled by
  similarity.
- Layer 3: a single distinct query ("YOU") marker with a CSS pulse animation.
- Axes hidden (PCA components are not interpretable units); tooltip shows
  patient id, similarity, and outcomes.

### 5.4 Side panel
- Outcome breakdown card: for each lens, `positive/total` and a small bar.
- Twin list: sorted by similarity; clicking/hovering a twin highlights its dot
  and opens a detail popover with:
  - real UPCI trajectory sparkline (recharts LineChart), and
  - "Matched on: <feature> (you 110 / twin 115), ..." (top-3 matched features).
- Outcome-lens toggle recolors twins (CKD / Delayed remission / Relapse).

### 5.5 Styling
- Dark "space" panel for the constellation to sell the wow factor, while
  remaining visually consistent with the existing Tailwind UI. Outcome colors
  reuse the dashboard's risk palette where sensible.

### 5.6 Safety
- Persistent inline note: "Case-based model association, not diagnosis or
  treatment guidance; requires clinical validation." Mirrors existing
  `SAFETY_NOTE` framing.

## 6. Components and boundaries

- `find_twins(patient_ref, inputs, n)` — pure service function; input =
  patient ref/inputs, output = the JSON payload above; depends only on existing
  loaders + a cached PCA/standardizer. Independently testable.
- `_twin_matrix()` / `_twin_pca()` / `_twin_stats()` — cached helpers for the
  standardized matrix, fitted PCA, and cohort mean/std.
- `/api/twins` route — thin adapter (parse body -> call `find_twins` -> jsonify),
  matching the existing `/api/predict` route style.
- `TwinsModule` React component — owns fetch + view state (active lens, selected
  twin); renders `Constellation` (chart) + `TwinPanel` (breakdown/list/detail).

## 7. Testing

- Backend: a pytest in the existing suite covering `find_twins()`:
  - returns exactly `n` twins (and clamps out-of-range `n`),
  - twins sorted by descending similarity, similarity in [0,100],
  - outcome breakdown counts match the twins' outcome flags,
  - query projection is finite; deterministic across runs (PCA random_state),
  - trajectory points map to the right months.
- Frontend: type-check via `tsc` build; manual smoke test that the module loads,
  the lens toggle recolors, and twin hover shows the trajectory.

## 8. Out of scope / future

- UMAP/t-SNE alternative projection, similarity feature weighting, twin
  treatment-regimen overlay, exportable twin report. Not in this iteration.
