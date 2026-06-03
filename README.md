# SLECare INNOVERSE Dashboard

Clinical/research dashboard prototype for Lupus Nephritis risk assessment. The dashboard is designed for clinician-researcher review of the LN cohort, individual CKD and delayed-remission risk prediction, explainability, high-risk patient prioritization, patient timeline review, global SHAP interpretation, and research-ready export summaries.

This is a decision-support prototype. It is not a diagnostic replacement, does not recommend treatment, and requires clinical validation before real-world deployment.

## Dashboard Modules

The sidebar should contain exactly these six modules:

1. Cohort Overview
2. Patient Risk Assessment
3. High-Risk Patients
4. Patient Timeline
5. Global Risk Drivers
6. Research Export

Do not add generic ML-demo modules such as model leaderboards, treatment recommendation engines, chatbots, appointment schedulers, or authentication unless explicitly requested.

## Tech Stack

- Frontend: React, TypeScript, Vite, Tailwind CSS, Recharts, Lucide React
- Backend: Python Flask
- Clinical-facing model: CatBoost only
- Explainability: local SHAP payloads for patient assessment and global SHAP image assets for cohort-level risk drivers

## Directory Guide

```text
.
|-- AGENTS.md                         # Codex/project guardrails for future contributors
|-- PROJECT_CONTEXT.md                # Product, clinical, and workflow context
|-- SPEC.md                           # Current dashboard specification
|-- PLANS.md                          # Implementation plan and task breakdown
|-- README.md                         # This handoff guide
|-- package.json                      # Frontend scripts and npm dependencies
|-- package-lock.json                 # Locked npm dependency versions
|-- index.html                        # Vite HTML entry point
|-- vite.config.ts                    # Vite config
|-- tsconfig.json                     # TypeScript config
|-- tailwind.config.js                # Tailwind config
|-- postcss.config.js                 # PostCSS config
|-- src/                              # React dashboard source
|   |-- App.tsx                       # Main six-module dashboard UI
|   |-- main.tsx                      # React entry point
|   |-- styles.css                    # Tailwind and app styling
|   |-- types.ts                      # Shared frontend types
|   `-- services/
|       `-- api.ts                    # Frontend API adapter and fallback logic
|-- backend/                          # Flask API and model/data services
|   |-- app.py                        # API routes
|   |-- services.py                   # Dataset, CatBoost, SHAP, and payload logic
|   `-- __init__.py
|-- data/                             # Anonymized LN dataset and optional processed CSVs
|   |-- ANONYMOUS DATA EXCEL LN RELAPSE_UKM.xlsx
|   |-- CB_CKD_cleaned_with_target.csv
|   `-- CB_Remission_cleaned_with_target.csv
|-- model/                            # Dashboard CatBoost models and feature lists
|   |-- catboost_ckd_model.pkl
|   |-- catboost_remission_model.pkl
|   |-- ckd_rfe_features.json
|   `-- remission_rfe_features.json
`-- assets/                           # Global SHAP plots used by the dashboard
    |-- ckd_barplot.png
    |-- remission_barplot.png
    `-- Dependence Plots/
        |-- CKD_CREAT.png
        |-- CKD_LA.png
        |-- CKD_RACE.png
        |-- Remission_CKD.png
        |-- Remission_GLOBAL.png
        |-- Remission_MONTH.png
        `-- Remission_RACE.png
```

## What To Upload To GitHub

Upload these files and folders:

- `AGENTS.md`
- `PROJECT_CONTEXT.md`
- `SPEC.md`
- `PLANS.md`
- `README.md`
- `package.json`
- `package-lock.json`
- `index.html`
- `vite.config.ts`
- `tsconfig.json`
- `tailwind.config.js`
- `postcss.config.js`
- `src/`
- `backend/`
- `assets/`
- `model/`
- `data/`, only if the data is approved for sharing in the target GitHub repository

Do not upload these generated or local-only folders/files:

- `node_modules/`
- `dist/`
- `backend/__pycache__/`
- `.venv/`, `venv/`, or other Python environments
- `.env` or files containing API keys/secrets
- local editor folders such as `.vscode/` or `.idea/`
- OS files such as `Thumbs.db` or `.DS_Store`

Important data note: the repository currently contains anonymized clinical data and trained model files. Before pushing to a public GitHub repository, confirm that `data/` and `model/` are permitted to be shared. If they are not approved for public sharing, keep those folders out of GitHub and provide teammates a private secure transfer instead. The app should keep graceful fallback behavior for missing files where possible.

## Local Setup

Install frontend dependencies:

```bash
npm install
```

Install backend dependencies in a Python environment:

```bash
python -m pip install flask pandas numpy shap catboost openpyxl
```

Run the backend API:

```bash
python -m backend.app
```

Run the frontend in another terminal:

```bash
npm run dev
```

Open the Vite URL shown in the terminal, usually:

```text
http://127.0.0.1:5173
```

The frontend expects the backend API at:

```text
http://127.0.0.1:5000
```

## Build

Build the frontend:

```bash
npm run build
```

## Backend API

- `GET /api/dashboard` returns cohort, patient, high-risk, timeline, SHAP asset, and export payloads.
- `POST /api/predict` accepts patient inputs and returns CKD and delayed-remission risk outputs with local SHAP-style explanations.
- `GET /api/assets/<asset_path>` serves SHAP image assets from `assets/`.

## Development Rules

- Use CatBoost only for clinical-facing CKD and delayed-remission predictions.
- Keep mock-safe logic clearly isolated and replaceable.
- The LLM layer, if added later, must only convert structured model/SHAP output into concise clinical language. It must not make predictions or treatment recommendations.
- Use cautious clinical wording: risk prediction, decision-support, may support earlier identification, not a diagnostic replacement.
- Preserve the six approved modules and avoid dashboard bloat.
