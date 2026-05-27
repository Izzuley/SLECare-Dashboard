import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  ClipboardCopy,
  Download,
  FileText,
  Gauge,
  LineChart as LineChartIcon,
  Loader2,
  Search,
  ShieldAlert,
  Sparkles,
  Stethoscope,
  Users,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { loadDashboard, predictPatient } from "./services/api";
import type { DashboardPayload, ModuleName, PatientRow, RiskCategory, RiskOutcome, RiskResult } from "./types";

const navIcons = [Users, Stethoscope, ShieldAlert, LineChartIcon, BarChart3, FileText];
const palette = ["#0f766e", "#2563eb", "#b45309", "#7c3aed", "#64748b"];
const importantFields = [
  "CREAT BASELINE",
  "ALBUMIN BASELINE",
  "C4 PRETX",
  "CHRONIC INDEX",
  "ACTIVE INDEX",
  "GLOBAL SCLEROSIS",
  "MONTH INTERVAL TO INDUCTION TX",
  "UPCI PRE TX",
  "CKD",
  "LA",
  "RACE",
  "ACE/ARB",
];

function pct(value: number) {
  return `${Math.round(value * 100)}%`;
}

function riskClass(category: RiskCategory) {
  if (category === "High") return "bg-rose-50 text-rose-700 border-rose-200";
  if (category === "Moderate") return "bg-amber-50 text-amber-700 border-amber-200";
  return "bg-emerald-50 text-emerald-700 border-emerald-200";
}

function StatCard({ label, value, detail }: { label: string; value: string | number; detail: string }) {
  return (
    <div className="panel p-5">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-2 text-3xl font-semibold text-ink">{value}</p>
      <p className="mt-2 text-sm text-slate-600">{detail}</p>
    </div>
  );
}

function OutcomeCard({ outcome }: { outcome: RiskOutcome }) {
  return (
    <div className="panel p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-slate-500">{outcome.targetEvent}</p>
          <h3 className="mt-1 text-2xl font-semibold text-ink">{pct(outcome.probability)}</h3>
        </div>
        <span className={`rounded-full border px-3 py-1 text-sm font-semibold ${riskClass(outcome.riskCategory)}`}>
          {outcome.riskCategory}
        </span>
      </div>
      <div className="mt-4 h-2 rounded-full bg-slate-100">
        <div className="h-2 rounded-full bg-clinical" style={{ width: `${Math.round(outcome.probability * 100)}%` }} />
      </div>
      <p className="mt-4 text-xs text-slate-500">{outcome.thresholdNotes}</p>
    </div>
  );
}

function RiskDrivers({ result }: { result: RiskResult }) {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {result.outcomes.map((outcome) => (
        <div key={outcome.target} className="panel p-5">
          <h3 className="text-lg font-semibold text-ink">{outcome.target} local explanation</h3>
          <p className="mt-1 text-sm text-slate-600">Local SHAP drivers are model associations for this patient, not causation.</p>
          <div className="mt-4 space-y-3">
            {outcome.topRiskDrivers.slice(0, 4).map((driver) => (
              <div key={`${outcome.target}-${driver.feature}`} className="rounded-md border border-slate-200 p-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="font-semibold text-ink">{driver.displayName}</p>
                    <p className="text-sm text-slate-600">{driver.clinicalMeaning}</p>
                  </div>
                  <span className="shrink-0 rounded-full bg-rose-50 px-2 py-1 text-xs font-semibold text-rose-700">
                    +{Math.abs(driver.shapValue).toFixed(3)}
                  </span>
                </div>
              </div>
            ))}
          </div>
          {outcome.protectiveFactors.length > 0 && (
            <div className="mt-4">
              <p className="text-sm font-semibold text-slate-600">Protective contributors</p>
              <p className="mt-1 text-sm text-slate-600">
                {outcome.protectiveFactors
                  .slice(0, 3)
                  .map((driver) => driver.displayName)
                  .join(", ")}
              </p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function CohortOverview({ data }: { data: DashboardPayload }) {
  const stats = data.cohort.summaryStats.slice(0, 6);
  return (
    <section className="space-y-6">
      <div className="section-heading">
        <div>
          <h1>Cohort Overview</h1>
          <p>LN cohort burden, outcome distribution, and selected clinical characteristics from the available dataset.</p>
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <StatCard label="Raw cohort records" value={data.cohort.rawRecords} detail="From the anonymised Excel workbook." />
        <StatCard label="CKD model-ready rows" value={data.cohort.ckdModelReadyRecords} detail="Rows aligned to CKD CatBoost features." />
        <StatCard
          label="Remission model-ready rows"
          value={data.cohort.remissionModelReadyRecords}
          detail="Rows aligned to delayed-remission CatBoost features."
        />
      </div>
      <div className="grid gap-5 lg:grid-cols-2">
        <ChartPanel title="CKD target distribution" data={data.cohort.ckdDistribution} />
        <ChartPanel title="Delayed-remission target distribution" data={data.cohort.delayedRemissionDistribution} />
        <ChartPanel title="Race code distribution" data={data.cohort.raceDistribution} />
        <ChartPanel title="Gender code distribution" data={data.cohort.genderDistribution} />
      </div>
      <div className="panel p-5">
        <h2 className="text-lg font-semibold text-ink">Clinical profile summary</h2>
        <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {stats.map((stat) => (
            <div key={stat.field} className="rounded-md border border-slate-200 bg-white p-4">
              <p className="text-sm font-semibold text-ink">{stat.label}</p>
              <p className="mt-1 text-sm text-slate-600">Median {stat.median} · mean {stat.mean}</p>
              <p className="mt-1 text-xs text-slate-500">{stat.available} records available</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function ChartPanel({ title, data }: { title: string; data: { name: string; value: number }[] }) {
  return (
    <div className="panel p-5">
      <h2 className="text-lg font-semibold text-ink">{title}</h2>
      <div className="mt-4 h-64">
        <ResponsiveContainer>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="value" radius={[6, 6, 0, 0]}>
              {data.map((_, index) => (
                <Cell key={index} fill={palette[index % palette.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function PatientRiskAssessment({
  data,
  selectedPatient,
  setSelectedPatientId,
}: {
  data: DashboardPayload;
  selectedPatient: PatientRow;
  setSelectedPatientId: (id: string) => void;
}) {
  const [inputs, setInputs] = useState<Record<string, number | string | null>>(selectedPatient.inputs);
  const [result, setResult] = useState<RiskResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setInputs(selectedPatient.inputs);
    setResult(null);
    setError("");
  }, [selectedPatient]);

  const fields = useMemo(() => importantFields.filter((field) => field in inputs), [inputs]);

  async function runPrediction() {
    setBusy(true);
    setError("");
    try {
      setResult(await predictPatient(selectedPatient.id, inputs));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setBusy(false);
    }
  }

  function updateField(field: string, value: string) {
    const parsed = value.trim() === "" ? null : Number(value);
    setInputs((current) => ({ ...current, [field]: Number.isNaN(parsed) ? value : parsed }));
  }

  return (
    <section className="space-y-6">
      <div className="section-heading">
        <div>
          <h1>Patient Risk Assessment</h1>
          <p>CatBoost CKD and delayed-remission risk prediction with local SHAP-style explanation.</p>
        </div>
        <button onClick={runPrediction} className="primary-button" disabled={busy}>
          {busy ? <Loader2 className="animate-spin" size={18} /> : <Gauge size={18} />}
          Generate
        </button>
      </div>
      <div className="panel p-5">
        <label className="label" htmlFor="patient-select">Sample patient</label>
        <select
          id="patient-select"
          className="input mt-2"
          value={selectedPatient.id}
          onChange={(event) => setSelectedPatientId(event.target.value)}
        >
          {data.patients.slice(0, 60).map((patient) => (
            <option key={patient.id} value={patient.id}>{patient.displayId}</option>
          ))}
        </select>
        <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {fields.map((field) => (
            <label key={field} className="block">
              <span className="label">{field}</span>
              <input className="input mt-1" value={String(inputs[field] ?? "")} onChange={(event) => updateField(field, event.target.value)} />
            </label>
          ))}
        </div>
        {result && result.inputValidation.warnings.length > 0 && (
          <p className="mt-4 rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
            {result.inputValidation.warnings[0]}
          </p>
        )}
        {error && <p className="mt-4 rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">{error}</p>}
      </div>
      {result && result.predictionKind === "full" ? (
        <>
          <div className="grid gap-5 lg:grid-cols-2">
            {result.outcomes.map((outcome) => <OutcomeCard key={outcome.target} outcome={outcome} />)}
          </div>
          <div className="panel p-5">
            <div className="flex items-center gap-2">
              <Sparkles size={18} className="text-clinical" />
              <h2 className="text-lg font-semibold text-ink">Clinician-friendly explanation</h2>
            </div>
            <p className="mt-3 text-slate-700">{result.llmExplanation.summary}</p>
            <p className="mt-3 text-sm font-semibold text-slate-600">{result.llmExplanation.safetyNote}</p>
          </div>
          <RiskDrivers result={result} />
        </>
      ) : (
        <div className="panel p-5">
          <h2 className="text-lg font-semibold text-ink">Ready for model inference</h2>
          <p className="mt-2 text-sm text-slate-600">
            Select a sample patient or edit fields, then click Generate to run full CatBoost + local SHAP prediction for this assessment.
          </p>
        </div>
      )}
    </section>
  );
}

function HighRiskPatients({ patients, onSelectAssessment, onSelectTimeline }: {
  patients: PatientRow[];
  onSelectAssessment: (id: string) => void;
  onSelectTimeline: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [risk, setRisk] = useState("All");
  const sorted = useMemo(() => {
    return [...patients]
      .filter((patient) => risk === "All" || patient.priority.urgency === risk)
      .filter((patient) => patient.displayId.toLowerCase().includes(query.toLowerCase()))
      .sort((a, b) => b.priority.combinedRisk - a.priority.combinedRisk)
      .slice(0, 60);
  }, [patients, query, risk]);

  return (
    <section className="space-y-6">
      <div className="section-heading">
        <div>
          <h1>High-Risk Patients</h1>
          <p>Prioritized review list based on CKD risk, delayed-remission risk, and local driver summaries.</p>
        </div>
      </div>
      <div className="panel p-4">
        <div className="grid gap-3 md:grid-cols-[1fr_220px]">
          <label className="relative">
            <Search className="pointer-events-none absolute left-3 top-3 text-slate-400" size={18} />
            <input className="input pl-10" placeholder="Search sample or patient ID" value={query} onChange={(event) => setQuery(event.target.value)} />
          </label>
          <select className="input" value={risk} onChange={(event) => setRisk(event.target.value)}>
            <option>All</option>
            <option>High</option>
            <option>Moderate</option>
            <option>Low</option>
          </select>
        </div>
      </div>
      <div className="panel overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
            <thead className="bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="px-4 py-3">Patient</th>
                <th className="px-4 py-3">CKD</th>
                <th className="px-4 py-3">Delayed remission</th>
                <th className="px-4 py-3">Priority</th>
                <th className="px-4 py-3">Main reason</th>
                <th className="px-4 py-3">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {sorted.map((patient) => {
                const ckd = patient.prediction.outcomes[0];
                const remission = patient.prediction.outcomes[1];
                return (
                  <tr key={patient.id} className="hover:bg-slate-50">
                    <td className="px-4 py-3 font-semibold text-ink">{patient.displayId}</td>
                    <td className="px-4 py-3">{pct(ckd.probability)} · {ckd.riskCategory}</td>
                    <td className="px-4 py-3">{pct(remission.probability)} · {remission.riskCategory}</td>
                    <td className="px-4 py-3">
                      <span className={`rounded-full border px-2 py-1 text-xs font-semibold ${riskClass(patient.priority.urgency)}`}>
                        {patient.priority.urgency}
                      </span>
                    </td>
                    <td className="max-w-sm px-4 py-3 text-slate-600">{patient.priority.mainReason}</td>
                    <td className="px-4 py-3">
                      <div className="flex gap-2">
                        <button className="secondary-button" onClick={() => onSelectAssessment(patient.id)}>Assessment</button>
                        <button className="secondary-button" onClick={() => onSelectTimeline(patient.id)}>Timeline</button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function PatientTimeline({ selectedPatient, patients, setSelectedPatientId }: {
  selectedPatient: PatientRow;
  patients: PatientRow[];
  setSelectedPatientId: (id: string) => void;
}) {
  return (
    <section className="space-y-6">
      <div className="section-heading">
        <div>
          <h1>Patient Timeline</h1>
          <p>Prototype longitudinal view using biopsy timing, induction interval, creatinine, UPCI, and remission markers where available.</p>
        </div>
      </div>
      <div className="panel p-5">
        <label className="label" htmlFor="timeline-select">Patient</label>
        <select id="timeline-select" className="input mt-2" value={selectedPatient.id} onChange={(event) => setSelectedPatientId(event.target.value)}>
          {patients.slice(0, 60).map((patient) => <option key={patient.id} value={patient.id}>{patient.displayId}</option>)}
        </select>
        <div className="mt-5 grid gap-4 md:grid-cols-4">
          <StatCard label="Biopsy date" value={selectedPatient.biopsyDate?.slice(0, 10) ?? "Unavailable"} detail="From raw cohort workbook." />
          <StatCard label="Active LN year" value={selectedPatient.yearActiveLn ?? "Unavailable"} detail="Recorded year field." />
          <StatCard label="Induction interval" value={`${selectedPatient.inductionIntervalMonths ?? "NA"} mo`} detail="Month interval to induction treatment." />
          <StatCard label="Latest creatinine marker" value={selectedPatient.latestMarker ?? "NA"} detail="24-month creatinine where present." />
        </div>
      </div>
      <div className="grid gap-5 lg:grid-cols-2">
        <TrendPanel title="UPCI progression" dataKey="upci" patient={selectedPatient} />
        <TrendPanel title="Creatinine progression" dataKey="creatinine" patient={selectedPatient} />
      </div>
      <div className="panel p-5">
        <h2 className="text-lg font-semibold text-ink">Timeline notes</h2>
        <p className="mt-2 text-sm text-slate-600">
          Some intermediate creatinine timepoints are not available in the source file, so this view uses baseline and 24-month creatinine with UPCI follow-up points.
          No treatment recommendation is generated.
        </p>
      </div>
    </section>
  );
}

function TrendPanel({ title, dataKey, patient }: { title: string; dataKey: "upci" | "creatinine"; patient: PatientRow }) {
  return (
    <div className="panel p-5">
      <h2 className="text-lg font-semibold text-ink">{title}</h2>
      <div className="mt-4 h-72">
        <ResponsiveContainer>
          <LineChart data={patient.timeline}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="label" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey={dataKey} stroke="#0f766e" strokeWidth={3} connectNulls dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function GlobalRiskDrivers({ data }: { data: DashboardPayload }) {
  const [target, setTarget] = useState<"ckd" | "remission">("ckd");
  const assets = data.shapAssets[target];
  return (
    <section className="space-y-6">
      <div className="section-heading">
        <div>
          <h1>Global Risk Drivers</h1>
          <p>Cohort-level SHAP assets for model behavior review. These are model associations, not causal findings.</p>
        </div>
        <div className="segmented">
          <button className={target === "ckd" ? "selected" : ""} onClick={() => setTarget("ckd")}>CKD</button>
          <button className={target === "remission" ? "selected" : ""} onClick={() => setTarget("remission")}>Delayed remission</button>
        </div>
      </div>
      <div className="panel p-5">
        <h2 className="text-lg font-semibold text-ink">Global SHAP bar plot</h2>
        <img className="mt-4 max-h-[560px] w-full rounded-md border border-slate-200 object-contain" src={assets.barPlot} alt={`${target} global SHAP bar plot`} />
      </div>
      <div className="grid gap-5 lg:grid-cols-2">
        {assets.dependencePlots.map((plot) => (
          <div key={plot.src} className="panel p-5">
            <h3 className="font-semibold text-ink">{plot.title}</h3>
            <img className="mt-3 w-full rounded-md border border-slate-200 object-contain" src={plot.src} alt={`${plot.title} SHAP dependence plot`} />
          </div>
        ))}
      </div>
      <div className="panel p-5 text-sm text-slate-700">
        {target === "ckd"
          ? "CKD global drivers highlighted for clinical review include baseline creatinine, race code, lupus anticoagulant, pretreatment C4, and chronicity index."
          : "Delayed-remission global drivers highlighted for clinical review include CKD marker, induction-treatment interval, race code, global sclerosis, ACE/ARB marker, and lupus anticoagulant."}
      </div>
    </section>
  );
}

function ResearchExport({ data }: { data: DashboardPayload }) {
  const highRiskCount = data.patients.filter((patient) => patient.priority.urgency === "High").length;
  const text = `SLECare INNOVERSE prototype summary

Cohort: ${data.cohort.rawRecords} raw LN records; ${data.cohort.ckdModelReadyRecords} CKD model-ready records; ${data.cohort.remissionModelReadyRecords} remission model-ready records.
Outcome distributions are based on available non-missing targets in the cleaned CSV files.
High-priority prototype review list: ${highRiskCount} sample rows classified as high combined risk using CatBoost risk outputs.
Global SHAP review: CKD drivers include baseline creatinine, race code, lupus anticoagulant, pretreatment C4, and chronicity index. Delayed-remission drivers include CKD marker, induction interval, race code, global sclerosis, ACE/ARB marker, and lupus anticoagulant.
Safety wording: This is a decision-support risk prediction prototype and requires clinical validation before real-world deployment.`;

  function downloadCsv() {
    const rows = ["patient,ckd_risk,delayed_remission_risk,priority,reason"].concat(
      data.patients.map((patient) => {
        const ckd = patient.prediction.outcomes[0];
        const remission = patient.prediction.outcomes[1];
        return [patient.id, ckd.probability, remission.probability, patient.priority.urgency, `"${patient.priority.mainReason}"`].join(",");
      }),
    );
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "slecare-risk-summary.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className="space-y-6">
      <div className="section-heading">
        <div>
          <h1>Research Export</h1>
          <p>Export-ready summaries for abstracts, reports, posters, and research review.</p>
        </div>
        <div className="flex gap-2">
          <button className="secondary-button" onClick={() => navigator.clipboard.writeText(text)}>
            <ClipboardCopy size={17} /> Copy
          </button>
          <button className="primary-button" onClick={downloadCsv}>
            <Download size={17} /> CSV
          </button>
        </div>
      </div>
      <div className="grid gap-5 lg:grid-cols-3">
        <StatCard label="High-priority rows" value={highRiskCount} detail="Prototype combined risk category." />
        <StatCard label="CKD SHAP plots" value={4} detail="One bar plot and three dependence plots." />
        <StatCard label="Remission SHAP plots" value={5} detail="One bar plot and four dependence plots." />
      </div>
      <div className="panel p-5">
        <h2 className="text-lg font-semibold text-ink">Export-ready summary</h2>
        <pre className="mt-4 whitespace-pre-wrap rounded-md border border-slate-200 bg-slate-50 p-4 text-sm leading-6 text-slate-700">{text}</pre>
      </div>
    </section>
  );
}

export default function App() {
  const [data, setData] = useState<DashboardPayload | null>(null);
  const [active, setActive] = useState<ModuleName>("Cohort Overview");
  const [selectedPatientId, setSelectedPatientId] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    loadDashboard()
      .then((payload) => {
        setData(payload);
        setSelectedPatientId(payload.patients[0]?.id ?? "");
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Unable to load dashboard."));
  }, []);

  const selectedPatient = useMemo(() => data?.patients.find((patient) => patient.id === selectedPatientId) ?? data?.patients[0], [data, selectedPatientId]);

  if (error) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-slate-100 p-6">
        <div className="panel max-w-lg p-6">
          <h1 className="text-xl font-semibold text-ink">Backend service required</h1>
          <p className="mt-2 text-slate-600">{error} Start the Flask API on port 5000, then reload the Vite app.</p>
        </div>
      </main>
    );
  }

  if (!data || !selectedPatient) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-slate-100">
        <Loader2 className="animate-spin text-clinical" size={32} />
      </main>
    );
  }

  function selectAndOpen(module: ModuleName, patientId: string) {
    setSelectedPatientId(patientId);
    setActive(module);
  }

  return (
    <div className="min-h-screen bg-slate-100 text-ink">
      <aside className="fixed inset-y-0 left-0 z-20 hidden w-72 border-r border-slate-200 bg-white px-4 py-5 lg:block">
        <div className="flex items-center gap-3 px-2">
          <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-clinical text-white">
            <Activity size={22} />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-500">SLECare</p>
            <h1 className="text-lg font-semibold text-ink">INNOVERSE</h1>
          </div>
        </div>
        <nav className="mt-8 space-y-1">
          {data.modules.map((module, index) => {
            const Icon = navIcons[index];
            return (
              <button key={module} className={`nav-item ${active === module ? "active" : ""}`} onClick={() => setActive(module)}>
                <Icon size={18} />
                {module}
              </button>
            );
          })}
        </nav>
        <div className="absolute bottom-5 left-4 right-4 rounded-md border border-teal-100 bg-teal-50 p-4 text-xs leading-5 text-teal-900">
          Decision-support prototype only. Not diagnostic, not treatment guidance, and requires clinical validation before real-world use.
        </div>
      </aside>
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/95 px-4 py-3 backdrop-blur lg:hidden">
        <select className="input" value={active} onChange={(event) => setActive(event.target.value as ModuleName)}>
          {data.modules.map((module) => <option key={module}>{module}</option>)}
        </select>
      </header>
      <main className="px-4 py-6 lg:ml-72 lg:px-8">
        {active === "Cohort Overview" && <CohortOverview data={data} />}
        {active === "Patient Risk Assessment" && (
          <PatientRiskAssessment data={data} selectedPatient={selectedPatient} setSelectedPatientId={setSelectedPatientId} />
        )}
        {active === "High-Risk Patients" && (
          <HighRiskPatients
            patients={data.patients}
            onSelectAssessment={(id) => selectAndOpen("Patient Risk Assessment", id)}
            onSelectTimeline={(id) => selectAndOpen("Patient Timeline", id)}
          />
        )}
        {active === "Patient Timeline" && (
          <PatientTimeline selectedPatient={selectedPatient} patients={data.patients} setSelectedPatientId={setSelectedPatientId} />
        )}
        {active === "Global Risk Drivers" && <GlobalRiskDrivers data={data} />}
        {active === "Research Export" && <ResearchExport data={data} />}
      </main>
    </div>
  );
}
