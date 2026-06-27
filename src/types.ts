export type ModuleName =
  | "Cohort Overview"
  | "Patient Risk Assessment"
  | "High-Risk Patients"
  | "Patient Timeline"
  | "Global Risk Drivers"
  | "Research Export"
  | "Patient Twins";

export type RiskCategory = "Low" | "Moderate" | "High";

export type ShapDriver = {
  feature: string;
  displayName: string;
  value: string | number | boolean | null;
  shapValue: number;
  direction: "increases risk" | "reduces risk";
  rank: number;
  clinicalMeaning: string;
  evidenceScope: string;
};

export type RiskOutcome = {
  target: "CKD" | "Delayed remission";
  targetEvent: string;
  modelFile: string;
  probability: number;
  riskCategory: RiskCategory;
  thresholdNotes: string;
  topRiskDrivers: ShapDriver[];
  protectiveFactors: ShapDriver[];
};

export type RiskResult = {
  schemaVersion: string;
  patientRef: string;
  generatedAt: string;
  predictionKind: "summary" | "full";
  predictionSource: "catboost" | "mock";
  shapSource: "local_shap" | "mock_shap";
  inputValidation: { missingRequiredFields: string[]; invalidFields?: string[]; warnings: string[] };
  outcomes: RiskOutcome[];
  llmExplanation: {
    summary: string;
    mainDrivers: string[];
    protectiveFactors: string[];
    safetyNote: string;
  };
};

export type TimelinePoint = {
  month: number;
  label: string;
  creatinine: number | null;
  upci: number | null;
};

export type PatientRow = {
  id: string;
  displayId: string;
  rawIndex: number;
  race: number | string | null;
  gender: number | string | null;
  ageAtInduction: number | null;
  biopsyDate: string | null;
  yearActiveLn: number | null;
  inductionIntervalMonths: number | null;
  latestMarker: number | null;
  inputs: Record<string, number | string | null>;
  timeline: TimelinePoint[];
  prediction: RiskResult;
  priority: {
    combinedRisk: number;
    urgency: RiskCategory;
    mainReason: string;
  };
};

export type SimulationDelta = {
  target: "CKD" | "Delayed remission";
  targetEvent: string;
  baselineProbability: number;
  simulatedProbability: number;
  delta: number;
  deltaPercentagePoints: number;
  baselineCategory: RiskCategory;
  simulatedCategory: RiskCategory;
  direction: "reduced" | "increased" | "unchanged";
};

export type SimulationResult = {
  schemaVersion: string;
  patientRef: string;
  generatedAt: string;
  modifiedInputs: Record<string, number | string | null>;
  baseline: RiskResult;
  simulated: RiskResult;
  deltas: SimulationDelta[];
  safetyNote: string;
};

export type ModifiableDriver = {
  feature: string;
  displayName: string;
  baselineValue: number | string | null;
  suggestedValue: number;
  perturbation: string;
  baselineRisk: number;
  simulatedRisk: number;
  riskReduction: number;
  riskReductionPct: number;
  confidence: "Low" | "Low-Moderate" | "Moderate";
  clinicalMeaning: string;
  evidenceScope: string;
  rank: number;
};

export type DriverEngineResult = {
  schemaVersion: string;
  patientRef: string;
  generatedAt: string;
  baselineRisk: Record<"CKD" | "Delayed remission", number>;
  rankings: Record<"CKD" | "Delayed remission", ModifiableDriver[]>;
  topDriver: Record<"CKD" | "Delayed remission", ModifiableDriver | null>;
  safetyNote: string;
};

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

export type DashboardPayload = {
  modules: ModuleName[];
  generatedAt: string;
  cohort: {
    rawRecords: number;
    ckdModelReadyRecords: number;
    remissionModelReadyRecords: number;
    ckdDistribution: { name: string; value: number }[];
    delayedRemissionDistribution: { name: string; value: number }[];
    raceDistribution: { name: string; value: number }[];
    genderDistribution: { name: string; value: number }[];
    summaryStats: { field: string; label: string; median: number; mean: number; available: number }[];
  };
  patients: PatientRow[];
  featureLists: { ckd: string[]; remission: string[] };
  shapAssets: {
    ckd: { barPlot: string; dependencePlots: { title: string; src: string }[] };
    remission: { barPlot: string; dependencePlots: { title: string; src: string }[] };
  };
};
