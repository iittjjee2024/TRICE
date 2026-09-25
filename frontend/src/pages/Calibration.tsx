import { useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Line,
  LineChart,
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
  Pill,
  Spinner,
  Table,
  Td,
  fmt,
  num,
} from "../components/ui";

export default function Calibration() {
  const { runId } = useRun();
  const q = useQuery({
    queryKey: ["calibration", runId],
    queryFn: () => api.calibration(runId!),
    enabled: !!runId,
  });

  if (!runId) return <Empty>No run selected.</Empty>;
  if (q.isLoading) return <Spinner label="Loading calibration" />;
  if (q.error) return <ErrorBox error={q.error} />;
  const groups = q.data!.groups;

  return (
    <div className="space-y-5">
      <Card
        title="Why calibration is load-bearing here"
        subtitle="The decision layer is only optimal if the probabilities it consumes are honest"
      >
        <p className="text-[13px] text-ink-300">
          The expected-F<sub>0.5</sub> rule computes{" "}
          <code className="mono">E[F | k]</code> from a Poisson-binomial over the candidate
          probabilities. If those probabilities are systematically off, the argmax over{" "}
          <code className="mono">k</code> lands in the wrong place even though the ranking is
          unchanged — a failure mode completely invisible in AUC. Calibration is therefore
          fitted per country on training pairs only, with a pooled fallback for groups that
          are small or, like <b className="text-ink-200">France</b>, entirely unseen in
          training.
        </p>
      </Card>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {groups.map((g) => (
          <div
            key={g.group}
            className="rounded-lg border border-ink-700 bg-ink-900/60 px-3 py-2.5"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-[12.5px] font-semibold text-ink-200">
                {g.group}
              </span>
              {g.is_overall ? (
                <Pill tone="accent">pooled</Pill>
              ) : (
                <Pill tone={g.has_own_curve ? "good" : "warn"}>
                  {g.has_own_curve ? "own curve" : "pooled fallback"}
                </Pill>
              )}
            </div>
            <div className="tnum mt-2 grid grid-cols-3 gap-2 text-[11.5px]">
              <div>
                <div className="text-ink-400">ECE</div>
                <div className="font-semibold text-ink-200">{fmt(g.ece, 4)}</div>
              </div>
              <div>
                <div className="text-ink-400">Brier</div>
                <div className="font-semibold text-ink-200">{fmt(g.brier, 4)}</div>
              </div>
              <div>
                <div className="text-ink-400">pairs</div>
                <div className="font-semibold text-ink-200">{num(g.n)}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {groups.map((g) => (
        <Card
          key={g.group}
          title={`Reliability — ${g.group}`}
          subtitle="Predicted probability vs observed match rate; the diagonal is perfect"
          right={<Pill>{num(g.n)} pairs</Pill>}
        >
          {!g.points.length ? (
            <Empty>No binned points for this group.</Empty>
          ) : (
            <div className="grid gap-4 lg:grid-cols-[1.3fr_1fr]">
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={g.points.map((p) => ({ ...p, ideal: p.p_pred }))}
                    margin={{ top: 8, right: 16, bottom: 4, left: -12 }}
                  >
                    <CartesianGrid stroke="#1f2632" />
                    <XAxis
                      dataKey="p_pred"
                      type="number"
                      domain={[0, 1]}
                      stroke="#6b7688"
                      tick={{ fontSize: 11 }}
                      tickFormatter={(v) => v.toFixed(1)}
                      label={{
                        value: "predicted probability",
                        position: "insideBottom",
                        offset: -2,
                        fill: "#6b7688",
                        fontSize: 11,
                      }}
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
                    <Line
                      type="monotone"
                      dataKey="ideal"
                      stroke="#2c3442"
                      strokeDasharray="4 4"
                      dot={false}
                      name="perfect"
                    />
                    <Line
                      type="monotone"
                      dataKey="p_true"
                      stroke="#3ddc97"
                      strokeWidth={2}
                      dot={{ r: 2.5 }}
                      name="observed"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <Table head={["Predicted", "Observed", "Gap", "Pairs"]} dense>
                {g.points.map((p) => (
                  <tr key={p.p_pred} className="border-b border-ink-800/70">
                    <Td dense className="tnum">
                      {fmt(p.p_pred, 4)}
                    </Td>
                    <Td dense className="tnum">
                      {fmt(p.p_true, 4)}
                    </Td>
                    <Td
                      dense
                      className={`tnum ${
                        Math.abs(p.p_pred - p.p_true) > 0.05 ? "text-warn" : "text-ink-400"
                      }`}
                    >
                      {(p.p_true - p.p_pred >= 0 ? "+" : "") +
                        (p.p_true - p.p_pred).toFixed(4)}
                    </Td>
                    <Td dense className="tnum text-ink-400">
                      {num(p.n)}
                    </Td>
                  </tr>
                ))}
              </Table>
            </div>
          )}
        </Card>
      ))}
    </div>
  );
}
