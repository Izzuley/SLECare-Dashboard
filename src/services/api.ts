import type {
  DashboardPayload,
  DriverEngineResult,
  RiskResult,
  SimulationResult,
  TwinsResult,
} from "../types";

export async function loadDashboard(): Promise<DashboardPayload> {
  const response = await fetch("/api/dashboard");
  if (!response.ok) {
    throw new Error("Dashboard data service is unavailable.");
  }
  return response.json();
}

export async function predictPatient(
  patientRef: string,
  inputs: Record<string, number | string | null>,
): Promise<RiskResult> {
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patientRef, inputs }),
  });
  if (!response.ok) {
    throw new Error("Prediction service is unavailable.");
  }
  return response.json();
}

export async function simulatePatient(
  patientRef: string,
  modifiedInputs: Record<string, number | string | null>,
  baselineInputs: Record<string, number | string | null>,
): Promise<SimulationResult> {
  const response = await fetch("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patientRef, modifiedInputs, baselineInputs }),
  });
  if (!response.ok) {
    throw new Error("Simulation service is unavailable.");
  }
  return response.json();
}

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

export async function loadTopDrivers(
  patientRef: string,
  baselineInputs: Record<string, number | string | null>,
): Promise<DriverEngineResult> {
  const response = await fetch("/api/drivers", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patientRef, baselineInputs }),
  });
  if (!response.ok) {
    throw new Error("Driver engine service is unavailable.");
  }
  return response.json();
}
