import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { useRun } from "../state/RunContext";
import {
  Card,
  Empty,
  ErrorBox,
  Flag,
  Pill,
  Spinner,
  Stat,
  Table,
  Td,
  fmt,
  num,
} from "../components/ui";

export default function Dashboard() {
  const { runs, runId, current, loading, error } = useRun();
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const ablation = useQuery({
    queryKey: ["ablation", runId],
    queryFn: () => api.ablation(runId!),
    enabled: !!runId,
  });

  if (loading) return <Spinner label="Loading runs" />;
  if (error) return <ErrorBox error={error} />;

  const h = current?.headline;
  const decisionGain =
    h?.val_macro_f05 != null && h?.best_threshold_f05 != null
      ? h.val_macro_f05 - h.best_threshold_f05
      : null;

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-6">
        <Stat
          label="Val macro F0.5"
          value={fmt(h?.val_macro_f05, 4)}
          tone="accent"
          big
          hint="held-out entities"
        />
        <Stat label="Macro precision" value={fmt(h?.val_macro_precision, 4)} big />
        <Stat label="Macro recall" value={fmt(h?.val_macro_recall, 4)} big />
        <Stat
          label="Blocking recall"
          value={fmt(h?.blocking_recall, 4)}
          hint="candidate-set ceiling"
        />
        <Stat
          label="Stage-1 AUC"
          value={fmt(h?.stage1_auc, 5)}
          hint={current?.use_stage2 ? "stage 2 adopted" : "stage 2 rejected"}
        />
        <Stat
          label="Mean |S|"
          value={fmt(h?.mean_k, 2)}
          hint="predictions per entity"
        />
      </div>

      <div className="grid gap-5 lg:grid-cols-3">
        <Card
          title="What this solution does"
          className="lg:col-span-2"
          subtitle="TRICE — tripartite competition, corroboration, calibration, expected-F selection"
        >
          <ol className="space-y-2.5 text-[13px] text-ink-300">
            <li>
              <b className="text-ink-200">Normalise.</b> Unicode folding plus a
              consonant-skeleton transform that maps Devanagari and Latin renderings of
              the same name to one string; DBA splitting, domain de-concatenation, legal
              suffix separation, order-free address component bags.
            </li>
            <li>
              <b className="text-ink-200">Mine variants.</b> Token aliases
              (<code className="mono">texas↔tx</code>,{" "}
              <code className="mono">mysore↔mysuru</code>, Devanagari state names)
              learned from ground-truth matched pairs rather than any external gazetteer.
            </li>
            <li>
              <b className="text-ink-200">Block.</b> Two independent IDF-weighted sparse
              channels (name, address) with per-namespace document-frequency caps, unioned.
            </li>
            <li>
              <b className="text-ink-200">Match.</b> Gradient-boosted trees on ~55
              pairwise features dominated by IDF-weighted rarity overlap.
            </li>
            <li>
              <b className="text-ink-200">Graph.</b> Column-softmax competition with a null
              dustbin (Source 1 is deduplicated, so each candidate has at most one parent)
              plus Source 2 ↔ Source 3 corroboration.
            </li>
            <li>
              <b className="text-ink-200">Decide.</b> Exact expected-F<sub>0.5</sub> set
              selection per entity via a Poisson-binomial DP — replaces threshold tuning.
            </li>
          </ol>
        </Card>

        <Card title="Environment">
          {health.isLoading ? (
            <Spinner />
          ) : health.error ? (
            <ErrorBox error={health.error} />
          ) : (
            <dl className="space-y-1.5 text-[12.5px]">
              <Row k="API" v={`v${health.data!.version}`} />
              <Row k="Python" v={health.data!.python} />
              <Row
                k="Record store"
                v={<Flag ok={health.data!.state.store_built} yes="built" no="missing" />}
              />
              <Row k="Runs" v={num(health.data!.state.n_runs)} />
              <Row
                k="Submission written"
                v={
                  <Flag
                    ok={health.data!.state.output_written}
                    yes="yes"
                    no="not yet"
                  />
                }
              />
              <div className="pt-2">
                <div className="mb-1 text-[10.5px] tracking-wider text-ink-400 uppercase">
                  Libraries
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(health.data!.capabilities).map(([k, v]) => (
                    <Pill key={k} tone={v ? "good" : "default"}>
                      {k}
                    </Pill>
                  ))}
                </div>
              </div>
            </dl>
          )}
        </Card>
      </div>

      <Card
        title="Ablation"
        subtitle="Every row scored on the same held-out validation entities"
        right={
          decisionGain != null && (
            <Pill tone={decisionGain > 0 ? "good" : "warn"}>
              decision layer {decisionGain >= 0 ? "+" : ""}
              {decisionGain.toFixed(5)} vs tuned threshold
            </Pill>
          )
        }
      >
        {!runId ? (
          <Empty>No run selected.</Empty>
        ) : ablation.isLoading ? (
          <Spinner />
        ) : ablation.error ? (
          <ErrorBox error={ablation.error} />
        ) : (
          <Table head={["Configuration", "Macro F0.5", "Δ vs TRICE", "Note"]}>
            {ablation.data!.rows.map((r) => (
              <tr key={r.config} className="border-b border-ink-800/70">
                <Td className="font-medium text-ink-200">{r.config}</Td>
                <Td className="tnum">{fmt(r.macro_f05, 5)}</Td>
                <Td
                  className={`tnum ${
                    r.delta_vs_expected_f == null
                      ? "text-ink-400"
                      : r.delta_vs_expected_f === 0
                        ? "text-ink-400"
                        : r.delta_vs_expected_f < 0
                          ? "text-good"
                          : "text-bad"
                  }`}
                >
                  {r.delta_vs_expected_f == null
                    ? "—"
                    : r.delta_vs_expected_f === 0
                      ? "baseline"
                      : (-r.delta_vs_expected_f).toFixed(5)}
                </Td>
                <Td className="text-ink-400">{r.note}</Td>
              </tr>
            ))}
          </Table>
        )}
        <p className="mt-3 text-[11.5px] text-ink-400">
          Δ is the gain TRICE achieves over that configuration. A negative Δ column value
          in a row means that row is worse than TRICE.
        </p>
      </Card>

      <Card title="All runs">
        {!runs.length ? (
          <Empty>
            No runs found. Train one with{" "}
            <code className="mono">python scripts/05_train.py</code>.
          </Empty>
        ) : (
          <Table
            head={[
              "Run",
              "Created",
              "Countries",
              "Entities",
              "Pairs",
              "F0.5",
              "P",
              "R",
              "Blocking R",
              "Stage 2",
              "Inference",
            ]}
          >
            {runs.map((r) => (
              <tr
                key={r.run_id}
                className={`border-b border-ink-800/70 ${
                  r.run_id === runId ? "bg-accent/5" : ""
                }`}
              >
                <Td className="mono font-medium text-ink-200">{r.run_id}</Td>
                <Td className="text-ink-400">{r.created?.replace("T", " ") ?? "—"}</Td>
                <Td>{r.countries.join(", ")}</Td>
                <Td className="tnum">{num(r.n_entities)}</Td>
                <Td className="tnum">{num(r.n_pairs)}</Td>
                <Td className="tnum font-semibold text-accent">
                  {fmt(r.headline.val_macro_f05, 4)}
                </Td>
                <Td className="tnum">{fmt(r.headline.val_macro_precision, 4)}</Td>
                <Td className="tnum">{fmt(r.headline.val_macro_recall, 4)}</Td>
                <Td className="tnum">{fmt(r.headline.blocking_recall, 4)}</Td>
                <Td>
                  <Flag ok={!!r.use_stage2} yes="adopted" no="rejected" />
                </Td>
                <Td>
                  {r.has_inference ? (
                    <Link to="/output" className="text-accent underline">
                      view
                    </Link>
                  ) : (
                    <span className="text-ink-400">—</span>
                  )}
                </Td>
              </tr>
            ))}
          </Table>
        )}
      </Card>
    </div>
  );
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-ink-400">{k}</dt>
      <dd className="mono text-right text-ink-200">{v}</dd>
    </div>
  );
}
