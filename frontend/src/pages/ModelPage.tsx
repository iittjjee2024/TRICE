import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import { useRun } from "../state/RunContext";
import {
  Card,
  Empty,
  ErrorBox,
  Flag,
  Spinner,
  Stat,
  Table,
  Td,
  fmt,
  num,
} from "../components/ui";

const FAMILY: [RegExp, string][] = [
  [/^name_idf|shared_.*idf|unshared/, "rarity"],
  [/^name_(ratio|token|partial|jaro)/, "name fuzzy"],
  [/^skel|nospace|acronym/, "structural"],
  [/^legal/, "legal suffix"],
  [/^addr/, "address"],
  [/^digit|house|postal/, "address numeric"],
  [/^compete/, "competition"],
  [/^corr_|other_source/, "corroboration"],
  [/^entity_/, "entity profile"],
  [/^probe|block_score|cand_/, "blocking context"],
  [/^p1/, "stage-1 score"],
  [/^cross_/, "cross-field"],
  [/^country|addr_empty/, "context"],
];

function family(name: string): string {
  for (const [re, label] of FAMILY) if (re.test(name)) return label;
  return "other";
}

const FAMILY_COLOR: Record<string, string> = {
  rarity: "#4ea1ff",
  "name fuzzy": "#6fb7ff",
  structural: "#8f7bff",
  "legal suffix": "#b57bff",
  address: "#3ddc97",
  "address numeric": "#5fe3b0",
  competition: "#ffb454",
  corroboration: "#ffd166",
  "entity profile": "#ff9f7a",
  "blocking context": "#97a1b2",
  "stage-1 score": "#ff6b6b",
  "cross-field": "#c3cad6",
  context: "#6b7688",
  other: "#48505e",
};

export default function ModelPage() {
  const { runId } = useRun();
  const q = useQuery({
    queryKey: ["features", runId],
    queryFn: () => api.features(runId!),
    enabled: !!runId,
  });

  if (!runId) return <Empty>No run selected.</Empty>;
  if (q.isLoading) return <Spinner label="Loading model report" />;
  if (q.error) return <ErrorBox error={q.error} />;
  const d = q.data!;

  const useStage2 = !!d.model.use_stage2;
  const imp = (useStage2 ? d.importances_stage2 : d.importances_stage1)
    .slice(0, 26)
    .map(([feature, importance]) => ({
      feature,
      importance,
      fam: family(feature),
    }));

  const famTotals = new Map<string, number>();
  for (const [f, v] of useStage2 ? d.importances_stage2 : d.importances_stage1) {
    const k = family(f);
    famTotals.set(k, (famTotals.get(k) ?? 0) + v);
  }
  const famRows = [...famTotals.entries()].sort((a, b) => b[1] - a[1]);

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-6">
        <Stat label="Model" value={d.model.kind ?? "—"} hint="gradient-boosted trees" />
        <Stat label="Stage-1 AUC" value={fmt(d.model.stage1?.auc, 5)} big />
        <Stat
          label="Stage-1 AP"
          value={fmt(d.model.stage1?.average_precision, 5)}
          hint="average precision"
        />
        <Stat label="Stage-2 AUC" value={fmt(d.model.stage2?.auc, 5)} big />
        <Stat
          label="Stage-2 AP"
          value={fmt(d.model.stage2?.average_precision, 5)}
        />
        <Stat
          label="Stage 2 adopted"
          value={<Flag ok={useStage2} yes="yes" no="no" />}
          hint="only if it beats stage 1 on val AP"
        />
      </div>

      <Card
        title="Model size and licensing"
        subtitle="Challenge rule: MIT/Apache-2.0 model, at most 8 billion parameters"
      >
        <div className="grid gap-3 sm:grid-cols-3">
          <Stat
            label="Stage-1 parameters"
            value={num(d.model.stage1_parameters)}
            hint="tree nodes × 2"
          />
          <Stat
            label="Stage-2 parameters"
            value={num(d.model.stage2_parameters)}
          />
          <Stat
            label="Headroom vs 8 B cap"
            value={`${(
              ((d.model.stage1_parameters ?? 0) + (d.model.stage2_parameters ?? 0)) /
              8e9 *
              100
            ).toExponential(1)}%`}
            tone="good"
            hint="comfortably inside the limit"
          />
        </div>
        <p className="mt-3 text-[12px] text-ink-300">
          {d.model.kind === "lightgbm" ? "LightGBM (MIT)" : "scikit-learn HistGradientBoosting (BSD-3-Clause)"}
          . No pretrained weights are downloaded and no external corpus is consulted — all
          corpus statistics (IDF, mined token aliases) come from the provided train and test
          files, which keeps the pipeline inside the fair-play rules.
        </p>
      </Card>

      <div className="grid gap-5 xl:grid-cols-[1.5fr_1fr]">
        <Card
          title={`Feature importance — stage ${useStage2 ? 2 : 1}`}
          subtitle="Normalised split gain, top 26"
        >
          <div style={{ height: Math.max(360, imp.length * 20) }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={imp}
                layout="vertical"
                margin={{ top: 4, right: 20, bottom: 4, left: 150 }}
              >
                <CartesianGrid stroke="#1f2632" horizontal={false} />
                <XAxis
                  type="number"
                  stroke="#6b7688"
                  tick={{ fontSize: 10 }}
                  tickFormatter={(v) => (v * 100).toFixed(0) + "%"}
                />
                <YAxis
                  type="category"
                  dataKey="feature"
                  stroke="#6b7688"
                  width={148}
                  tick={{ fontSize: 10.5 }}
                />
                <Tooltip
                  contentStyle={{
                    background: "#10141c",
                    border: "1px solid #2c3442",
                    fontSize: 12,
                  }}
                  formatter={(v: number, _n, p) =>
                    [`${(v * 100).toFixed(2)}%`, (p as any).payload.fam]
                  }
                />
                <Bar dataKey="importance" radius={[0, 3, 3, 0]}>
                  {imp.map((r) => (
                    <Cell key={r.feature} fill={FAMILY_COLOR[r.fam]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Importance by feature family" subtitle="Where the signal lives">
          <Table head={["Family", "Share"]}>
            {famRows.map(([f, v]) => (
              <tr key={f} className="border-b border-ink-800/70">
                <Td>
                  <span className="inline-flex items-center gap-2">
                    <span
                      aria-hidden="true"
                      className="inline-block h-2.5 w-2.5 rounded-sm"
                      style={{ background: FAMILY_COLOR[f] }}
                    />
                    {f}
                  </span>
                </Td>
                <Td className="tnum">{(v * 100).toFixed(2)}%</Td>
              </tr>
            ))}
          </Table>
          <p className="mt-3 text-[11.5px] text-ink-400">
            Rarity features dominate by design: a shared uncommon token is near-proof of a
            match, a shared <code className="mono">services</code> is worth almost nothing.
            IDF computed over the provided corpus measures this without any external data.
          </p>
        </Card>
      </div>

      <Card
        title="Single-feature separation"
        subtitle="Positive/negative means and standalone AUC on validation pairs"
      >
        <Table head={["Feature", "Family", "Positive mean", "Negative mean", "AUC"]}>
          {d.separation.map((s) => (
            <tr key={s.feature} className="border-b border-ink-800/70">
              <Td className="mono text-ink-200">{s.feature}</Td>
              <Td className="text-ink-400">{family(s.feature)}</Td>
              <Td className="tnum text-good">{fmt(s.pos_mean, 4)}</Td>
              <Td className="tnum text-ink-300">{fmt(s.neg_mean, 4)}</Td>
              <Td className="tnum font-medium">{fmt(s.auc, 4)}</Td>
            </tr>
          ))}
        </Table>
      </Card>
    </div>
  );
}
