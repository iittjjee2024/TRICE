import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type SimulateRequest } from "../api/client";
import { useRun } from "../state/RunContext";
import {
  Card,
  Empty,
  ErrorBox,
  Labeled,
  Pill,
  Spinner,
  Stat,
  Table,
  Td,
  btnCls,
  fmt,
  num,
  selectCls,
} from "../components/ui";

const DEFAULTS: SimulateRequest = {
  rule: "expected_f",
  beta: 0.5,
  prune_epsilon: 0.01,
  max_emit: 25,
  global_threshold: 0.5,
  fixed_k: 3,
  probability_power: 1.0,
  use_missing_mass: true,
  disjointness_repair: true,
  use_stage1_probabilities: false,
};

/** Debounce so dragging a slider issues one request, not forty. */
function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

const VERDICT_TONE: Record<string, string> = {
  exact: "text-good",
  singleton_ok: "text-good",
  partial: "text-warn",
  missed_all: "text-warn",
  false_merge_on_singleton: "text-bad",
  false_merge_on_matched: "text-bad",
};

export default function DecisionTuner() {
  const { runId } = useRun();
  const [req, setReq] = useState<SimulateRequest>(DEFAULTS);
  const debounced = useDebounced(req, 220);

  const stored = useQuery({
    queryKey: ["decision", runId],
    queryFn: () => api.decision(runId!),
    enabled: !!runId,
  });

  const sim = useQuery({
    queryKey: ["simulate", runId, debounced],
    queryFn: () => api.simulate(runId!, debounced),
    enabled: !!runId,
    placeholderData: (prev) => prev,
  });

  const baseline = useQuery({
    queryKey: ["simulate-baseline", runId],
    queryFn: () =>
      api.simulate(runId!, { ...DEFAULTS, rule: "global_threshold", global_threshold: 0.5 }),
    enabled: !!runId,
  });

  const set = <K extends keyof SimulateRequest>(k: K, v: SimulateRequest[K]) =>
    setReq((r) => ({ ...r, [k]: v }));

  const kHist = useMemo(() => {
    const d = sim.data;
    if (!d) return [];
    const pred = d.k_histogram;
    const truth = d.truth_size_histogram;
    const keys = new Set([...Object.keys(pred), ...Object.keys(truth)]);
    return [...keys]
      .map(Number)
      .sort((a, b) => a - b)
      .filter((k) => k <= 12)
      .map((k) => ({
        k,
        predicted: pred[String(k)] ?? 0,
        truth: truth[String(k)] ?? 0,
      }));
  }, [sim.data]);

  if (!runId) return <Empty>No run selected.</Empty>;
  if (stored.isLoading) return <Spinner label="Loading decision report" />;
  if (stored.error) return <ErrorBox error={stored.error} />;

  const s = sim.data?.score;
  const b = baseline.data?.score;
  const delta = s && b ? s.macro_f05 - b.macro_f05 : null;

  return (
    <div className="space-y-5">
      <Card
        title="The decision layer"
        subtitle="F0.5(S,T) = 1.25·|S∩T| / (0.25·|T| + |S|) — the denominator depends only on set sizes, which makes the optimal set computable exactly"
      >
        <p className="text-[13px] text-ink-300">
          For each entity the rule evaluates{" "}
          <code className="mono">E[F | top-k]</code> for every{" "}
          <code className="mono">k = 0…n</code> using a Poisson-binomial over the calibrated
          candidate probabilities, and takes the argmax. Selecting the top-k by probability
          is provably optimal for a fixed k, so only <code className="mono">n+1</code> sets
          need checking rather than <code className="mono">2ⁿ</code> — verified against
          exhaustive search in <code className="mono">scripts/test_decide.py</code>. The
          singleton decision is not special-cased: <code className="mono">k = 0</code> gives{" "}
          <code className="mono">E[F|0] = Π(1−pᵢ)</code>.
        </p>
      </Card>

      <div className="grid gap-5 xl:grid-cols-[320px_1fr]">
        <Card title="Parameters" subtitle="Re-solves instantly; no re-scoring needed">
          <div className="space-y-3.5">
            <Labeled label="Decision rule">
              <select
                className={selectCls}
                value={req.rule}
                onChange={(e) => set("rule", e.target.value as SimulateRequest["rule"])}
              >
                <option value="expected_f">expected_f (TRICE)</option>
                <option value="global_threshold">global_threshold</option>
                <option value="top1">top1</option>
                <option value="topk">topk (fixed cardinality)</option>
              </select>
            </Labeled>

            {req.rule === "global_threshold" && (
              <Labeled
                label={`Threshold — ${req.global_threshold.toFixed(2)}`}
                hint="the parameter the expected-F rule removes"
              >
                <input
                  type="range"
                  min={0.02}
                  max={0.98}
                  step={0.01}
                  value={req.global_threshold}
                  onChange={(e) => set("global_threshold", Number(e.target.value))}
                  className="w-full accent-[#4ea1ff]"
                />
              </Labeled>
            )}

            {req.rule === "topk" && (
              <Labeled label={`Fixed k — ${req.fixed_k}`}>
                <input
                  type="range"
                  min={1}
                  max={10}
                  step={1}
                  value={req.fixed_k}
                  onChange={(e) => set("fixed_k", Number(e.target.value))}
                  className="w-full accent-[#4ea1ff]"
                />
              </Labeled>
            )}

            <Labeled
              label={`β — ${req.beta.toFixed(2)}`}
              hint="0.5 is the leaderboard metric; raise it to see the rule become recall-hungry"
            >
              <input
                type="range"
                min={0.1}
                max={2}
                step={0.05}
                value={req.beta}
                onChange={(e) => set("beta", Number(e.target.value))}
                className="w-full accent-[#4ea1ff]"
              />
            </Labeled>

            <Labeled
              label={`Prune ε — ${req.prune_epsilon.toFixed(3)}`}
              hint="probabilities below this fold into the missing-mass term"
            >
              <input
                type="range"
                min={0}
                max={0.2}
                step={0.005}
                value={req.prune_epsilon}
                onChange={(e) => set("prune_epsilon", Number(e.target.value))}
                className="w-full accent-[#4ea1ff]"
              />
            </Labeled>

            <Labeled
              label={`Probability power γ — ${req.probability_power.toFixed(2)}`}
              hint="p → p^γ; the escape hatch if independence proves optimistic"
            >
              <input
                type="range"
                min={0.4}
                max={2.5}
                step={0.05}
                value={req.probability_power}
                onChange={(e) => set("probability_power", Number(e.target.value))}
                className="w-full accent-[#4ea1ff]"
              />
            </Labeled>

            <Labeled label={`Max |S| — ${req.max_emit}`}>
              <input
                type="range"
                min={1}
                max={48}
                step={1}
                value={req.max_emit}
                onChange={(e) => set("max_emit", Number(e.target.value))}
                className="w-full accent-[#4ea1ff]"
              />
            </Labeled>

            <fieldset className="space-y-2 border-t border-ink-700 pt-3">
              <legend className="sr-only">Toggles</legend>
              {(
                [
                  ["use_missing_mass", "Missing-mass term (blocking recall coupling)"],
                  ["disjointness_repair", "Disjointness repair (one parent per candidate)"],
                  ["use_stage1_probabilities", "Use stage-1 probabilities instead"],
                ] as [keyof SimulateRequest, string][]
              ).map(([k, label]) => (
                <label key={k} className="flex items-start gap-2 text-[12px]">
                  <input
                    type="checkbox"
                    checked={Boolean(req[k])}
                    onChange={(e) => set(k, e.target.checked as never)}
                    className="mt-0.5 accent-[#4ea1ff]"
                  />
                  <span className="text-ink-300">{label}</span>
                </label>
              ))}
            </fieldset>

            <button className={btnCls} onClick={() => setReq(DEFAULTS)}>
              Reset to chosen configuration
            </button>
          </div>
        </Card>

        <div className="space-y-5">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
            <Stat
              label="Macro F0.5"
              value={fmt(s?.macro_f05, 5)}
              tone="accent"
              big
              hint={sim.isFetching ? "updating…" : `${num(sim.data?.n_entities)} entities`}
            />
            <Stat label="Macro precision" value={fmt(s?.macro_precision, 4)} big />
            <Stat label="Macro recall" value={fmt(s?.macro_recall, 4)} big />
            <Stat label="Mean |S|" value={fmt(s?.mean_k, 3)} />
            <Stat
              label="Δ vs threshold 0.5"
              value={delta == null ? "—" : (delta >= 0 ? "+" : "") + delta.toFixed(5)}
              tone={delta == null ? "default" : delta >= 0 ? "good" : "bad"}
            />
          </div>

          {sim.error && <ErrorBox error={sim.error} />}

          <Card
            title="Predicted vs true set-size distribution"
            subtitle="How well the chosen cardinality tracks reality — the actual job of this layer"
          >
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={kHist} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
                  <CartesianGrid stroke="#1f2632" />
                  <XAxis
                    dataKey="k"
                    stroke="#6b7688"
                    tick={{ fontSize: 11 }}
                    label={{
                      value: "set size",
                      position: "insideBottom",
                      offset: -2,
                      fill: "#6b7688",
                      fontSize: 11,
                    }}
                  />
                  <YAxis stroke="#6b7688" tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      background: "#10141c",
                      border: "1px solid #2c3442",
                      fontSize: 12,
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="truth" name="true |T|" fill="#2c3442" radius={[3, 3, 0, 0]} />
                  <Bar
                    dataKey="predicted"
                    name="predicted |S|"
                    fill="#4ea1ff"
                    radius={[3, 3, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <div className="grid gap-5 lg:grid-cols-2">
            <Card
              title="Is E[F] honest?"
              subtitle="Predicted expected score vs realised score, binned"
            >
              {!sim.data?.ev_calibration.length ? (
                <Empty>Only produced by the expected_f rule.</Empty>
              ) : (
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={sim.data.ev_calibration}
                      margin={{ top: 8, right: 14, bottom: 4, left: -12 }}
                    >
                      <CartesianGrid stroke="#1f2632" />
                      <XAxis
                        dataKey="predicted_ef"
                        type="number"
                        domain={[0, 1]}
                        stroke="#6b7688"
                        tick={{ fontSize: 11 }}
                        tickFormatter={(v) => v.toFixed(1)}
                      />
                      <YAxis
                        domain={[0, 1]}
                        stroke="#6b7688"
                        tick={{ fontSize: 11 }}
                        tickFormatter={(v) => v.toFixed(1)}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#10141c",
                          border: "1px solid #2c3442",
                          fontSize: 12,
                        }}
                        formatter={(v: number) => v.toFixed(4)}
                      />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line
                        type="monotone"
                        dataKey="predicted_ef"
                        stroke="#2c3442"
                        strokeDasharray="4 4"
                        dot={false}
                        name="perfect"
                      />
                      <Line
                        type="monotone"
                        dataKey="realised_f"
                        stroke="#3ddc97"
                        strokeWidth={2}
                        name="realised F"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
              <p className="mt-2 text-[11.5px] text-ink-400">
                Deviation here means the independence assumption in the Poisson-binomial is
                optimistic; the γ slider is the single-parameter correction.
              </p>
            </Card>

            <Card title="Outcome breakdown" subtitle="Per-entity verdicts">
              <Table head={["Verdict", "Entities", "Share"]}>
                {Object.entries(sim.data?.verdicts ?? {})
                  .sort((a, b) => b[1] - a[1])
                  .map(([k, v]) => (
                    <tr key={k} className="border-b border-ink-800/70">
                      <Td className={VERDICT_TONE[k] ?? "text-ink-300"}>
                        {k.replace(/_/g, " ")}
                      </Td>
                      <Td className="tnum">{num(v)}</Td>
                      <Td className="tnum text-ink-400">
                        {sim.data
                          ? ((v / sim.data.n_entities) * 100).toFixed(2) + "%"
                          : "—"}
                      </Td>
                    </tr>
                  ))}
              </Table>
            </Card>
          </div>

          <Card title="By country" subtitle="France carries 15% of the test set">
            <Table
              head={["Country", "Entities", "Macro F0.5", "Precision", "Recall", "Mean |S|"]}
            >
              {(sim.data?.by_country ?? []).map((c) => (
                <tr key={c.country} className="border-b border-ink-800/70">
                  <Td className="font-medium text-ink-200">{c.country}</Td>
                  <Td className="tnum">{num(c.n)}</Td>
                  <Td className="tnum font-semibold text-accent">{fmt(c.macro_f05, 5)}</Td>
                  <Td className="tnum">{fmt(c.macro_precision, 4)}</Td>
                  <Td className="tnum">{fmt(c.macro_recall, 4)}</Td>
                  <Td className="tnum">{fmt(c.mean_k, 3)}</Td>
                </tr>
              ))}
            </Table>
          </Card>
        </div>
      </div>

      <Card
        title="Rule comparison from training"
        subtitle="Full validation split, computed during the run"
        right={
          stored.data?.best_threshold_rule && (
            <Pill>best tuned threshold: {stored.data.best_threshold_rule}</Pill>
          )
        }
      >
        <Table head={["Rule", "Macro F0.5", "Precision", "Recall", "Mean |S|"]}>
          {(stored.data?.comparison ?? []).map((r: any) => (
            <tr
              key={r.rule}
              className={`border-b border-ink-800/70 ${
                r.rule === "expected_f" ? "bg-accent/5" : ""
              }`}
            >
              <Td className="mono text-ink-200">{r.rule}</Td>
              <Td className="tnum font-semibold">{fmt(r.macro_f05, 5)}</Td>
              <Td className="tnum">{fmt(r.macro_precision, 4)}</Td>
              <Td className="tnum">{fmt(r.macro_recall, 4)}</Td>
              <Td className="tnum">{fmt(r.mean_k, 3)}</Td>
            </tr>
          ))}
        </Table>
      </Card>
    </div>
  );
}
