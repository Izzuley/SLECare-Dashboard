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
                        {point.outcomes?.[lens] === null
                          ? "unknown"
                          : point.outcomes?.[lens]
                            ? "yes"
                            : "no"}
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
                onClick={(point) => setSelectedTwinId((point as unknown as TwinPoint).id)}
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
                data={[
                  {
                    ...data.query,
                    id: "YOU",
                    displayId: `${selectedPatient.displayId} (this patient)`,
                  },
                ]}
                shape={(props: { cx?: number; cy?: number }) => (
                  <circle
                    className="twin-you-marker"
                    cx={props.cx}
                    cy={props.cy}
                    r={8}
                    fill="#facc15"
                    stroke="#fff"
                    strokeWidth={2}
                  />
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
            <p className="text-xs text-slate-500">
              {LENS_LABEL[lens]} among twins with a known outcome.
            </p>
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
                    {m.displayName}: you {String(m.queryValue ?? "NA")} / twin{" "}
                    {String(m.twinValue ?? "NA")}
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
