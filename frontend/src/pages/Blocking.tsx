import { useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
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
  Spinner,
  Stat,
  Table,
  Td,
  fmt,
  num,
} from "../components/ui";

export default function Blocking() {
  const { runId } = useRun();
  const q = useQuery({
    queryKey: ["blocking", runId],
    queryFn: () => api.blocking(runId!),
    enabled: !!runId,
  });

  if (!runId) return <Empty>No run selected.</Empty>;
  if (q.isLoading) return <Spinner label="Loading blocking report" />;
  if (q.error) return <ErrorBox error={q.error} />;
  const d = q.data!;

  const achieved = d.overall_macro_recall ?? 0;
  const ceilingAt = d.f05_ceiling_curve.reduce(
    (best, p) =>
      Math.abs(p.recall - achieved) < Math.abs(best.recall - achieved) ? p : best,
    d.f05_ceiling_curve[0],
  );

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat
          label="Overall macro recall"
          value={fmt(d.overall_macro_recall, 4)}
          tone="accent"
          big
          hint="mean over country partitions"
        />
        <Stat
          label="Achieved precision"
          value={fmt(d.achieved_precision, 4)}
          hint="from the decision stage"
        />
        <Stat
          label="F0.5 ceiling at this recall"
          value={fmt(ceilingAt?.f05_ceiling, 4)}
          hint="if the matcher were perfect on retrieved candidates"
        />
        <Stat
          label="Partitions"
          value={num(d.per_country.length)}
          hint="one index per country label"
        />
      </div>

      <Card
        title="Why recall matters less than it looks"
        subtitle="β = 0.5 discounts recall: F0.5 = 1.25·P·R / (0.25·P + R)"
      >
        <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={d.f05_ceiling_curve}
                margin={{ top: 8, right: 16, bottom: 4, left: -12 }}
              >
                <CartesianGrid stroke="#1f2632" />
                <XAxis
                  dataKey="recall"
                  stroke="#6b7688"
                  tick={{ fontSize: 11 }}
                  label={{
                    value: "blocking macro recall",
                    position: "insideBottom",
                    offset: -2,
                    fill: "#6b7688",
                    fontSize: 11,
                  }}
                />
                <YAxis
                  stroke="#6b7688"
                  tick={{ fontSize: 11 }}
                  domain={[0.6, 1]}
                  tickFormatter={(v) => v.toFixed(2)}
                />
                <Tooltip
                  contentStyle={{
                    background: "#10141c",
                    border: "1px solid #2c3442",
                    fontSize: 12,
                  }}
                  formatter={(v: number) => v.toFixed(4)}
                />
                <Line
                  type="monotone"
                  dataKey="f05_ceiling"
                  stroke="#4ea1ff"
                  dot={false}
                  strokeWidth={2}
                  name="F0.5 ceiling"
                />
                {ceilingAt && (
                  <ReferenceDot
                    x={ceilingAt.recall}
                    y={ceilingAt.f05_ceiling}
                    r={5}
                    fill="#3ddc97"
                    stroke="none"
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="text-[12.5px] text-ink-300">
            <p>
              The marker is where this run sits. Because precision is weighted twice as
              heavily as recall, moving blocking recall from 0.86 to 0.90 lifts the ceiling
              by only about two points, while a single false merge costs an entire entity.
            </p>
            <p className="mt-2">
              That said, with the matcher at precision {fmt(d.achieved_precision, 3)} rather
              than 1.0, recall has more headroom left than precision does — which is why the
              candidate cap was raised to 48 rather than tightened.
            </p>
            <Table head={["Recall", "F0.5 ceiling"]} dense>
              {d.f05_ceiling_curve
                .filter((_, i) => i % 5 === 0)
                .map((p) => (
                  <tr key={p.recall} className="border-b border-ink-800/70">
                    <Td dense className="tnum">
                      {p.recall.toFixed(2)}
                    </Td>
                    <Td dense className="tnum">
                      {p.f05_ceiling.toFixed(4)}
                    </Td>
                  </tr>
                ))}
            </Table>
          </div>
        </div>
      </Card>

      <Card
        title="Recall by country partition"
        subtitle="Macro recall bounds the achievable macro F0.5 for that partition"
      >
        <Table
          head={[
            "Country",
            "Macro recall",
            "Micro recall",
            "True links",
            "Found",
            "Entities at full recall",
            "Entities at zero recall",
          ]}
        >
          {d.per_country.map((c) => (
            <tr key={c.country} className="border-b border-ink-800/70">
              <Td className="font-medium text-ink-200">{c.country}</Td>
              <Td className="tnum font-semibold text-accent">
                {fmt(c.macro_recall, 4)}
              </Td>
              <Td className="tnum">{fmt(c.micro_recall, 4)}</Td>
              <Td className="tnum">{num(c.n_true_links)}</Td>
              <Td className="tnum">{num(c.n_found_links)}</Td>
              <Td className="tnum">{num(c.entities_full_recall)}</Td>
              <Td className="tnum text-warn">{num(c.entities_zero_recall)}</Td>
            </tr>
          ))}
        </Table>
      </Card>

      <Card
        title="Channel cost and contribution"
        subtitle="Two independent channels; the union is far stronger than either alone"
      >
        <Table
          head={[
            "Country",
            "Channel",
            "Vocabulary",
            "Index nnz",
            "Index GB",
            "Pairs retrieved",
            "Vocab s",
            "Build s",
            "Query s",
          ]}
        >
          {d.per_country.flatMap((c) =>
            c.channels.map((ch) => (
              <tr key={`${c.country}-${ch.channel}`} className="border-b border-ink-800/70">
                <Td className="text-ink-400">{c.country}</Td>
                <Td className="font-medium text-ink-200">{ch.channel}</Td>
                <Td className="tnum">{num(ch.vocab)}</Td>
                <Td className="tnum">{num(ch.nnz)}</Td>
                <Td className="tnum">{fmt(ch.index_gb, 2)}</Td>
                <Td className="tnum">{num(ch.pairs)}</Td>
                <Td className="tnum text-ink-400">{fmt(ch.seconds?.vocab, 0)}</Td>
                <Td className="tnum text-ink-400">{fmt(ch.seconds?.build, 0)}</Td>
                <Td className="tnum text-ink-400">{fmt(ch.seconds?.query, 0)}</Td>
              </tr>
            )),
          )}
        </Table>
        <p className="mt-3 text-[11.5px] text-ink-400">
          Retrieval cost is Σ<sub>t</sub> df<sub>query</sub>(t)·df<sub>index</sub>(t), so the
          per-namespace document-frequency caps are the primary performance control. Capping
          skeleton n-grams at df 1,200 while leaving name tokens at 4,000 cut estimated
          full-test query time from 165 min to 12 min.
        </p>
      </Card>

      {!!d.sweeps.length && (
        <Card
          title="Standalone blocking sweeps"
          subtitle="From scripts/04_blocking_eval.py"
        >
          <ul className="space-y-1 text-[12.5px]">
            {d.sweeps.map((s) => (
              <li key={s.file} className="mono text-ink-400">
                {s.file}
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}
