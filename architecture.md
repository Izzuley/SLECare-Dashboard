# SLECare INNOVERSE Dashboard Architecture

## 1. Purpose and Scope
SLECare is a clinician-researcher dashboard prototype for Lupus Nephritis decision-support and cohort review. It provides six fixed modules and uses CatBoost models for CKD and delayed-remission risk estimation.

The system is intentionally bounded to:
- cohort analytics
- patient risk assessment
- explainability display
- high-risk triage
- timeline visualization
- research summary export

It explicitly does not provide diagnosis or treatment recommendation.

## 2. Architecture at a Glance

```mermaid
flowchart LR
  A["React + Vite Frontend\nsrc/main.tsx, src/App.tsx"] -->|"/api/dashboard, /api/predict"| B["Flask API\nbackend/app.py"]
  B --> C["Service Layer\nbackend/services.py"]
  C --> D[("data files (.xlsx, .csv)")]
  C --> E[("model files (.pkl, *_features.json)")]
  B --> F["/api/assets/<path>"]
  F --> G[("asset images (.png)")]
```

## 3. Graphify Knowledge-Graph Findings (Source of Architectural Orientation)
Graph snapshot from [graphify-out/GRAPH_REPORT.md](./graphify-out/GRAPH_REPORT.md):
- build date: 2026-05-27
- corpus: 19 files, about 17,487 words
- graph size: 199 nodes, 321 edges, 16 communities
- extraction quality: 100% EXTRACTED (0% INFERRED, 0% AMBIGUOUS)
- graph commit fingerprint: `fbe8b1d8` (matches current HEAD)

High-centrality runtime nodes:
- `predict_patient()`
- `get_dashboard_payload()`
- `_patient_rows()`
- `_prepared_frame()`

Graphify confirmed key backend call edges:
- `dashboard()` -> `get_dashboard_payload()`
- `predict()` -> `predict_patient()`

These align with runtime routes in [backend/app.py](./backend/app.py).

## 4. Codebase Structure

```text
SLECare-Dashboard/
  src/                    React UI, module rendering, API client, TS contracts
  backend/                Flask routes + service/business logic
  data/                   Raw cohort source + cleaned model-ready datasets
  model/                  CatBoost binaries + selected feature lists
  assets/                 SHAP global plots and dependence plots
  graphify-out/           Generated architecture graph + report artifacts
```

## 5. Frontend Layer

### 5.1 Entry and App Shell
- [src/main.tsx](./src/main.tsx) mounts `App` in React StrictMode.
- [src/App.tsx](./src/App.tsx) contains the full application shell and all six module views.

### 5.2 Navigation and Module System
- Module names are part of a strict union type in [src/types.ts](./src/types.ts), preventing unsupported module labels.
- Sidebar and mobile select both render `data.modules` returned by backend.
- Active module state controls conditional rendering for:
  1. Cohort Overview
  2. Patient Risk Assessment
  3. High-Risk Patients
  4. Patient Timeline
  5. Global Risk Drivers
  6. Research Export

### 5.3 Frontend-Backend Integration
- [src/services/api.ts](./src/services/api.ts) defines only two network operations:
  - `loadDashboard()` -> `GET /api/dashboard`
  - `predictPatient(patientRef, inputs)` -> `POST /api/predict`
- [vite.config.ts](./vite.config.ts) proxies `/api` to `http://127.0.0.1:5000`, so frontend code can use relative API paths during development.

### 5.4 Type Contracts
[src/types.ts](./src/types.ts) defines the API contract shapes:
- `DashboardPayload`: module list, cohort stats/distributions, patient rows, feature lists, SHAP asset URLs
- `RiskResult`: prediction metadata (`predictionKind`, `predictionSource`, `shapSource`), outcomes, warnings, explanation summary
- `PatientRow`: patient-level fused object used by multiple modules (inputs, timeline, prediction, priority)

## 6. Backend Layer

### 6.1 API Surface
[backend/app.py](./backend/app.py) exposes:
- `GET /api/dashboard` -> full dashboard payload
- `POST /api/predict` -> on-demand prediction for edited/current patient inputs
- `GET /api/assets/<path>` -> static serving for SHAP images

### 6.2 Service Core
[backend/services.py](./backend/services.py) is the single business-logic module and contains:
- data/model/asset root constants
- module list constant (`MODULES`)
- feature label and clinical wording dictionaries
- data/model loaders (`_raw_df`, `_ckd_df`, `_remission_df`, `_ckd_model`, `_remission_model`)
- prediction pipeline (`_prepared_frame`, `_predict_one`, `predict_patient`)
- SHAP local explanation pipeline (`_explainer`, `_local_shap`, `_driver_payload`)
- patient table/timeline construction (`_patient_rows`, `_timeline`)
- full payload assembly (`get_dashboard_payload`)

### 6.3 Caching Strategy
There are two caching layers:
- `@lru_cache` on file/model/explainer loaders and payload builders (process-lifetime memoization)
- manual in-memory LRU-like cache (`OrderedDict`) for prediction responses keyed by canonicalized input payload, capped at 256 entries

This reduces repeated model loading and repeat prediction cost.

## 7. Data and Model Assets

### 7.1 Data Sources
- Raw Excel: `data/ANONYMOUS DATA EXCEL LN RELAPSE_UKM.xlsx`
- Cleaned CKD CSV: `data/CB_CKD_cleaned_with_target.csv`
- Cleaned remission CSV: `data/CB_Remission_cleaned_with_target.csv`

### 7.2 Model and Features
- CKD model: `model/catboost_ckd_model.pkl`
- Remission model: `model/catboost_remission_model.pkl`
- Feature lists:
  - `model/ckd_rfe_features.json`
  - `model/remission_rfe_features.json`

### 7.3 Explainability Assets
Global SHAP bar/dependence images live under `assets/` and are served via `/api/assets/*` URLs assembled in `get_dashboard_payload()`.

## 8. Core Runtime Processes

### 8.1 Process A: Application Boot and Initial Load
1. Frontend boots `App`.
2. `App` calls `loadDashboard()` on mount.
3. Flask route `/api/dashboard` calls `get_dashboard_payload()`.
4. Service loads cached data/models as needed, builds cohort stats + patient rows + SHAP asset links.
5. Frontend stores payload and selects first patient as default context.

### 8.2 Process B: Patient Risk Assessment (Interactive)
1. User selects/edits input fields in Patient Risk Assessment.
2. Frontend posts `{ patientRef, inputs }` to `/api/predict`.
3. Backend `predict_patient()`:
   - builds normalized feature frames for CKD and remission
   - imputes missing fields using cohort medians
   - applies categorical casting from CatBoost metadata
   - gets probabilities from both models
   - computes local SHAP values and ranks top risk/protective drivers
   - generates clinician-friendly summary text (`explain_payload`)
4. Frontend renders outcome cards, safety note, and driver panels.

### 8.3 Process C: High-Risk Triage Generation
1. During dashboard payload generation, `_patient_rows()` iterates patients.
2. It calls `predict_patient()` for each row to attach per-patient prediction.
3. Combined risk is computed as mean of CKD and remission probabilities.
4. Urgency category is derived from thresholds (`Low < 0.34`, `Moderate 0.34-0.66`, `High >= 0.67`).
5. Frontend table filters/searches/sorts these precomputed rows.

### 8.4 Process D: Timeline Assembly
1. `_timeline()` maps longitudinal columns into month-based points (`0, 3, 6, 12, 18, 24`).
2. Missing intermediate creatinine points stay `null`; UPCI is populated where available.
3. Frontend renders line charts for UPCI and creatinine trends.

### 8.5 Process E: Global Driver Visualization
1. Backend includes SHAP image URLs in payload.
2. Frontend switches between CKD and remission asset bundles.
3. Images are rendered directly from `/api/assets/...` endpoints.

### 8.6 Process F: Research Export
1. Frontend synthesizes a text summary from payload metrics.
2. User can copy summary to clipboard.
3. CSV export is generated client-side from patient prediction rows.

## 9. Clinical Safety and Product Guardrails
Safety constraints are enforced by both content and architecture:
- fixed module scope (no diagnosis/treatment module)
- risk outputs explicitly presented as decision-support
- SHAP interpretation framed as association, not causation
- explicit safety note included in prediction and shell UI

## 10. Non-Functional Characteristics
- Runtime style: monolith frontend + monolith Flask backend (simple deployment for prototype)
- Performance: cache-heavy in-memory strategy, no external DB
- State model: frontend local state only (`useState`, `useMemo`, `useEffect`)
- API model: small REST surface with stable JSON contracts

## 11. Extension Points
Natural evolutions without major redesign:
- move `backend/services.py` into domain modules (`data`, `inference`, `explainability`, `payloads`)
- add schema validation for `/api/predict` request body
- add persistence layer for audit logs/versioned predictions
- add authentication/authorization only if scope expands beyond prototype usage

## 12. Operational Notes
- Dev startup:
  - backend: `python -m backend.app` (port 5000)
  - frontend: `npm run dev` (port 5173)
- The frontend depends on backend availability; otherwise it shows a service-required error screen.
- Graph maintenance: run `graphify update .` after code changes to keep architectural graph current.
