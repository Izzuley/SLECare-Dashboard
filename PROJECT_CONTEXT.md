# PROJECT_CONTEXT.md — SLECare INNOVERSE Dashboard

## Project Name

SLECare INNOVERSE Dashboard

## Project Purpose

SLECare is an explainable AI clinical decision-support system for Malaysian Lupus Nephritis patients.

The system predicts two clinically important outcomes:

1. Chronic Kidney Disease risk
2. Delayed remission risk

The INNOVERSE dashboard prototype extends the original FYP by turning the prediction engine into a clinician-researcher workbench that supports patient assessment, risk explanation, high-risk prioritization, disease progression review, and research export.

## Intended User

Primary user:

- Prof. Dr. Syahrul Sazliyana Shaharir

User type:

- clinician
- researcher
- Lupus Nephritis domain expert

The dashboard must make sense to her work, not to machine learning developers.

She likely needs the dashboard to:

1. Review LN cohort burden.
2. Identify high-risk patients earlier.
3. Understand why a patient is high risk.
4. Prioritize patients for closer monitoring.
5. Review disease progression over time.
6. Reuse cohort findings for research papers, abstracts, and reports.

## Important Framing

This is not a generic machine learning dashboard.

This is not a model leaderboard.

This is not a toy predictor.

This is a comprehensive clinical/research dashboard prototype showing how SLECare could support real Lupus Nephritis clinical research workflow.

## Current Directory Structure

Expected current folder:

```text
Innoverse Dashboard/

assets/
- ckd_barplot.png
- remission_barplot.png
- Dependence Plots/
  - CKD_CREAT.png
  - CKD_LA.png
  - CKD_RACE.png
  - Remission_CKD.png
  - Remission_GLOBAL.png
  - Remission_MONTH.png
  - Remission_RACE.png

data/
- ANONYMOUS DATA EXCEL LN RELAPSE_UKM.xlsx
- optional: CB_CKD_cleaned_with_target.csv
- optional: CB_Remission_cleaned_with_target.csv

model/
- catboost_ckd_model.pkl
- catboost_remission_model.pkl
- ckd_rfe_features.json
- remission_rfe_features.json
```

## Optional Files to Add

If available, these files are useful:

`data/CB_CKD_cleaned_with_target.csv`

- RFE-selected and preprocessed CKD dataset with CKD target.

`data/CB_Remission_cleaned_with_target.csv`

- RFE-selected and preprocessed remission dataset with `CR_12_MTH_PRED_10` target.

These are useful for faster prototype prediction/demo logic and model-ready patient samples.

## Deployed Model Decision

Dashboard deployment uses CatBoost only.

CKD model:

- `model/catboost_ckd_model.pkl`

Delayed remission model:

- `model/catboost_remission_model.pkl`

Feature lists:

- `model/ckd_rfe_features.json`
- `model/remission_rfe_features.json`

Other models and ensembles are not part of the dashboard user workflow.

Reason:
The clinician user cares about the best practical model balance:

- performance
- interpretability
- stability
- clinical usefulness

CatBoost is the deployed dashboard model because it supports SHAP-based explanation and was selected as the strongest practical clinical-facing model.

## Dashboard Modules

### 1. Cohort Overview

Purpose:
Show the overall Lupus Nephritis cohort and disease burden.

Useful for:

- understanding the dataset population
- viewing CKD burden
- viewing delayed remission burden
- viewing race/gender/age distributions
- seeing high-level clinical/research context

Should include:

- total patients/episodes
- CKD distribution
- delayed remission distribution
- demographic charts
- key cohort statistics

### 2. Patient Risk Assessment

Purpose:
Assess one patient and explain the individual prediction.

This is the core clinical module.

Flow:

```text
Patient input or selected sample patient
→ CatBoost prediction
→ local SHAP explanation
→ SHAP payload
→ LLM-generated clinician-friendly explanation
```

Should include:

- CKD risk prediction
- delayed remission risk prediction
- risk probability
- low/moderate/high risk category
- key patient values
- top patient-specific risk drivers
- top protective factors where available
- concise AI-generated explanation

Example explanation:

> This patient is predicted to have high CKD risk mainly because baseline creatinine is elevated, lupus anticoagulant is positive, and chronic index is high. These factors suggest higher kidney involvement, so closer monitoring may be needed.

The LLM does not make predictions.
The LLM only explains CatBoost + SHAP output.

### 3. High-Risk Patients

Purpose:
Help the clinician prioritize which patients need attention first.

Useful for:

- reducing manual review burden
- identifying high-risk patients faster
- filtering patients by risk category or clinical markers

Should include:

- ranked high-risk patient list
- CKD risk
- delayed remission risk
- urgency flag
- main reason for flagging
- filters/search

Example:

> Patient X flagged because of high CKD risk, elevated creatinine, positive LA, and high chronicity index.

### 4. Patient Timeline

Purpose:
Show longitudinal patient progression.

Useful because Lupus Nephritis management is time-based and follow-up matters.

Should include:

- diagnosis / LN onset / biopsy / induction treatment timing
- baseline, 3M, 6M, 12M markers where available
- remission status
- relapse indicators
- kidney marker trends such as creatinine, albumin, UPCI, C3/C4 if available

If the raw data does not support a complete timeline, create a clean prototype timeline using available date/month fields and mock-safe fallback logic.

### 5. Global Risk Drivers

Purpose:
Show cohort-level model behavior.

This module uses global SHAP.

Should include:

- CKD global SHAP bar plot
- remission global SHAP bar plot
- CKD dependence plots
- remission dependence plots
- plain-language explanation of key cohort-level risk factors

Available assets:

- `assets/ckd_barplot.png`
- `assets/remission_barplot.png`
- `assets/Dependence Plots/*.png`

Clinical drivers to highlight:

CKD:

- `CREAT BASELINE`
- `RACE`
- `LA`
- `C4 PRETX`
- `CHRONIC INDEX`

Delayed remission:

- `CKD`
- `MONTH INTERVAL TO INDUCTION TX`
- `RACE`
- `GLOBAL SCLEROSIS`
- `ACE/ARB`
- `LA`

### 6. Research Export

Purpose:
Support the clinician-researcher workflow.

Useful for:

- abstracts
- papers
- posters
- grant reports
- presentations

Should include:

- cohort summary
- Table 1-style demographic/clinical summary
- outcome distribution
- top predictors
- SHAP findings summary
- export-ready text
- AI research summary / abstract support

The LLM can generate research-support text from:

- cohort statistics
- model outputs
- global SHAP findings

Do not fabricate results.
Only summarize available data and clearly label prototype/mock summaries when needed.

## Tech Stack

Frontend:

- React
- TypeScript
- Vite
- Tailwind CSS
- Recharts
- Lucide React

Backend:

- Python FastAPI or Flask if needed

Prediction:

- CatBoost only

Explainability:

- SHAP
- Local SHAP for patient prediction
- Global SHAP plots for cohort-level explanation

AI Layer:

- GPT / LLM API
- converts SHAP payload into clinician-friendly explanation
- converts cohort/model/SHAP findings into research summary text

Data:

- pandas
- NumPy
- Excel / CSV files

Storage:

- CSV/JSON or browser local state for prototype
- SQLite/PostgreSQL for future version

Export:

- CSV
- PDF
- Word/report text if feasible

Package manager:

- npm

## System Data Flow

Intended real flow:

```text
Clinician enters/selects patient
→ system validates fields
→ system passes model-ready features to CatBoost CKD/remission models
→ system produces risk probability and class
→ system calculates local SHAP or mock SHAP payload
→ system passes SHAP payload to LLM explanation layer
→ system displays concise clinician-friendly explanation
→ system stores or keeps result for later review
```

Prototype acceptable flow:

```text
User selects sample patient or enters simplified patient input
→ app uses available model files if integration works
→ otherwise uses deterministic mock prediction based on sample data
→ app displays realistic output format
→ app displays SHAP-payload-style explanation
→ app shows global SHAP assets
```

Mock logic must be isolated and clearly replaceable.

## SHAP Payload Concept

The SHAP payload is structured data passed to the LLM.

Example:

```json
{
  "target": "CKD",
  "prediction": "High risk",
  "probability": 0.78,
  "top_risk_drivers": [
    {
      "feature": "CREAT BASELINE",
      "value": 130,
      "direction": "increases risk",
      "clinical_meaning": "higher baseline creatinine suggests poorer kidney function"
    },
    {
      "feature": "LA",
      "value": 1,
      "direction": "increases risk",
      "clinical_meaning": "positive lupus anticoagulant is associated with higher risk in the model"
    }
  ],
  "protective_factors": [
    {
      "feature": "ALBUMIN BASELINE",
      "value": 38,
      "direction": "reduces risk",
      "clinical_meaning": "better albumin level may indicate better clinical status"
    }
  ]
}
```

The LLM should turn this into concise explanation text.

## Clinical Safety Constraints

The dashboard must not:

- diagnose SLE
- prescribe treatment
- recommend medication
- replace clinical judgment
- claim real-world clinical validation
- claim confirmed cost savings without data

The dashboard may say:

- predicts risk
- supports early identification
- assists review
- helps prioritize monitoring
- provides explainable decision-support
- supports research reporting

## UI/UX Requirements

Style:

- modern
- professional
- clean
- clinical
- not childish
- not cluttered

Navigation:
Use sidebar with exactly six modules:

1. Cohort Overview
2. Patient Risk Assessment
3. High-Risk Patients
4. Patient Timeline
5. Global Risk Drivers
6. Research Export

Design should include:

- cards for key stats
- charts for cohort overview
- tables for high-risk patients
- clean form/selectors for patient assessment
- SHAP images in Global Risk Drivers
- concise explanation panels
- export buttons or export-ready text sections

## What Not to Build

Do not build:

- standalone model benchmarking page
- generic chatbot
- treatment recommendation module
- fake cost calculator
- appointment scheduler
- authentication system
- hospital-wide EMR
- unnecessary animations
- excessive settings pages
- model leaderboard designed for ML developers

## Definition of Done

The prototype is acceptable when:

1. The app runs locally.
2. Six approved modules exist.
3. Cohort Overview reads the available dataset or a safe processed version.
4. Patient Risk Assessment supports sample patient selection or input.
5. CKD and delayed-remission risk outputs are displayed.
6. Local explanation/SHAP-payload-style explanation is displayed.
7. High-Risk Patients page ranks or filters patients.
8. Patient Timeline page shows longitudinal/progression view.
9. Global Risk Drivers page displays existing SHAP assets.
10. Research Export page generates export-ready summaries.
11. Code is modular and future-backend friendly.
12. No unnecessary model benchmarking module is added.
