import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
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
  Pill,
  Spinner,
  Table,
  Td,
  btnCls,
  fmt,
  inputCls,
  num,
  selectCls,
} from "../components/ui";

const VERDICT_TONE: Record<string, "good" | "warn" | "bad" | "default"> = {
  exact: "good",
  singleton_ok: "good",
  partial: "warn",
  missed_all: "warn",
  false_merge_on_singleton: "bad",
  false_merge_on_matched: "bad",
};

export default function Entities() {
  const { runId } = useRun();
  const [page, setPage] = useState(1);
  const [filter, setFilter] = useState("all");
  const [country, setCountry] = useState("");
  const [q, setQ] = useState("");
  const [sort, setSort] = useState("f05");
  const [ascending, setAscending] = useState(true);
  const [selected, setSelected] = useState<number | null>(null);

  const list = useQuery({
    queryKey: ["entities", runId, page, filter, country, q, sort, ascending],
    queryFn: () =>
      api.entities(runId!, {
        page,
        page_size: 40,
        filter,
        sort,
        ascending,
        ...(country ? { country } : {}),
        ...(q ? { q } : {}),
      }),
    enabled: !!runId,
  });

  const detail = useQuery({
    queryKey: ["entity", runId, selected],
    queryFn: () => api.entity(runId!, selected!),
    enabled: !!runId && selected !== null,
  });

  if (!runId) return <Empty>No run selected.</Empty>;

  return (
    <div className="space-y-5">
      <Card
        title="Entity explorer"
        subtitle="Macro averaging concentrates penalties in individual entities — this is where to read them"
      >
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <label className="block">
            <span className="mb-1 block text-[11px] text-ink-300">Outcome</span>
            <select
              className={selectCls}
              value={filter}
              onChange={(e) => {
                setFilter(e.target.value);
                setPage(1);
              }}
            >
              {(list.data?.filters ?? ["all"]).map((f) => (
                <option key={f} value={f}>
                  {f.replace(/_/g, " ")}
                </option>
              ))}
            </select>
          </label>
          <label className="block">
            <span className="mb-1 block text-[11px] text-ink-300">Country</span>
            <select
              className={selectCls}
              value={country}
              onChange={(e) => {
                setCountry(e.target.value);
                setPage(1);
              }}
            >
              <option value="">all</option>
              {(list.data?.countries ?? []).map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label className="block">
            <span className="mb-1 block text-[11px] text-ink-300">Entity id contains</span>
            <input
              className={inputCls}
              value={q}
              placeholder="S1-…"
              onChange={(e) => {
                setQ(e.target.value);
                setPage(1);
              }}
            />
          </label>
          <label className="block">
            <span className="mb-1 block text-[11px] text-ink-300">Sort by</span>
            <select
              className={selectCls}
              value={sort}
              onChange={(e) => setSort(e.target.value)}
            >
              <option value="f05">realised F0.5</option>
              <option value="n_cand">candidate count</option>
              <option value="k_star">predicted size</option>
              <option value="entity">entity order</option>
            </select>
          </label>
          <label className="flex items-end gap-2 pb-0.5">
            <input
              type="checkbox"
              checked={ascending}
              onChange={(e) => setAscending(e.target.checked)}
              className="accent-[#4ea1ff]"
            />
            <span className="text-[12px] text-ink-300">ascending (worst first)</span>
          </label>
        </div>
      </Card>

      {list.isLoading ? (
        <Spinner label="Loading entities" />
      ) : list.error ? (
        <ErrorBox error={list.error} />
      ) : (
        <Card
          title={`${num(list.data!.total)} entities match`}
          right={
            <div className="flex items-center gap-2">
              <button
                className={btnCls}
                disabled={page <= 1}
                onClick={() => setPage((p) => p - 1)}
              >
                ← prev
              </button>
              <span className="tnum text-[12px] text-ink-400">page {page}</span>
              <button
                className={btnCls}
                disabled={page * 40 >= list.data!.total}
                onClick={() => setPage((p) => p + 1)}
              >
                next →
              </button>
            </div>
          }
        >
          <Table
            head={[
              "Entity",
              "Country",
              "Cands",
              "|S|",
              "|T|",
              "TP",
              "E[F]",
              "F0.5",
              "Outcome",
              "",
            ]}
          >
            {list.data!.rows.map((r) => (
              <tr
                key={r.index}
                className={`border-b border-ink-800/70 ${
                  selected === r.index ? "bg-accent/5" : ""
                }`}
              >
                <Td className="mono text-ink-200">{r.entity_id}</Td>
                <Td className="text-ink-400">{r.country}</Td>
                <Td className="tnum">{r.n_candidates}</Td>
                <Td className="tnum">{r.k_star}</Td>
                <Td className="tnum">{r.truth_size}</Td>
                <Td className="tnum">{r.tp}</Td>
                <Td className="tnum text-ink-400">{fmt(r.expected_f, 3)}</Td>
                <Td className="tnum font-semibold">{fmt(r.realised_f, 4)}</Td>
                <Td>
                  <Pill tone={VERDICT_TONE[r.verdict] ?? "default"}>
                    {r.verdict.replace(/_/g, " ")}
                  </Pill>
                </Td>
                <Td>
                  <button
                    className="text-accent underline"
                    onClick={() => setSelected(r.index)}
                  >
                    inspect
                  </button>
                </Td>
              </tr>
            ))}
          </Table>
        </Card>
      )}

      {selected !== null && (
        <Card
          title="Entity detail"
          subtitle={detail.data?.note}
          right={
            <button className={btnCls} onClick={() => setSelected(null)}>
              close
            </button>
          }
        >
          {detail.isLoading ? (
            <Spinner />
          ) : detail.error ? (
            <ErrorBox error={detail.error} />
          ) : (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-2 text-[12.5px]">
                <span className="mono font-semibold text-ink-200">
                  {detail.data!.entity_id}
                </span>
                <Pill>{detail.data!.country}</Pill>
                <Pill tone={VERDICT_TONE[detail.data!.verdict] ?? "default"}>
                  {detail.data!.verdict.replace(/_/g, " ")}
                </Pill>
                <span className="text-ink-400">
                  candidates {detail.data!.n_candidates} · predicted{" "}
                  {detail.data!.k_star} · true {detail.data!.truth_size} · realised F0.5{" "}
                  <b className="text-ink-200">{fmt(detail.data!.realised_f, 4)}</b> ·
                  E[F] {fmt(detail.data!.expected_f, 4)} · missing-mass{" "}
                  {fmt(detail.data!.missing_mass, 3)}
                </span>
              </div>

              <div className="grid gap-4 xl:grid-cols-[1fr_1.3fr]">
                <div>
                  <h3 className="mb-2 text-[11px] tracking-wider text-ink-400 uppercase">
                    E[F | k] curve
                  </h3>
                  <div className="h-56">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={detail.data!.ev_curve}
                        margin={{ top: 8, right: 14, bottom: 4, left: -14 }}
                      >
                        <CartesianGrid stroke="#1f2632" />
                        <XAxis
                          dataKey="k"
                          stroke="#6b7688"
                          tick={{ fontSize: 11 }}
                          label={{
                            value: "k (predictions emitted)",
                            position: "insideBottom",
                            offset: -2,
                            fill: "#6b7688",
                            fontSize: 11,
                          }}
                        />
                        <YAxis
                          stroke="#6b7688"
                          tick={{ fontSize: 11 }}
                          domain={[0, 1]}
                          tickFormatter={(v) => v.toFixed(1)}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "#10141c",
                            border: "1px solid #2c3442",
                            fontSize: 12,
                          }}
                          formatter={(v: number) => v.toFixed(5)}
                        />
                        <ReferenceLine
                          x={detail.data!.ev_curve_k_star}
                          stroke="#3ddc97"
                          strokeDasharray="3 3"
                          label={{
                            value: `k*=${detail.data!.ev_curve_k_star}`,
                            fill: "#3ddc97",
                            fontSize: 11,
                            position: "top",
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="expected_f"
                          stroke="#4ea1ff"
                          strokeWidth={2}
                          dot={{ r: 2 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="mt-1 text-[11px] text-ink-400">
                    The peak is the chosen set size. A flat curve means the entity is
                    genuinely ambiguous; a sharp peak means the evidence is decisive.
                  </p>
                </div>

                <div>
                  <h3 className="mb-2 text-[11px] tracking-wider text-ink-400 uppercase">
                    Candidates
                  </h3>
                  <Table
                    head={["Candidate", "Src", "p", "p stage-1", "Block", "Emitted", "Truth"]}
                    dense
                  >
                    {detail.data!.candidates.map((c) => (
                      <tr
                        key={c.entity_id}
                        className={`border-b border-ink-800/70 ${
                          c.selected && !c.is_true
                            ? "bg-bad/10"
                            : !c.selected && c.is_true
                              ? "bg-warn/10"
                              : ""
                        }`}
                      >
                        <Td dense className="mono text-ink-200">
                          {c.entity_id}
                        </Td>
                        <Td dense className="text-ink-400">
                          S{c.source}
                        </Td>
                        <Td dense className="tnum font-semibold">
                          {fmt(c.p_calibrated, 4)}
                        </Td>
                        <Td dense className="tnum text-ink-400">
                          {fmt(c.p_stage1, 4)}
                        </Td>
                        <Td dense className="tnum text-ink-400">
                          {fmt(c.block_score, 3)}
                        </Td>
                        <Td dense>
                          <Flag ok={c.selected} yes="yes" no="no" />
                        </Td>
                        <Td dense>
                          <Flag ok={c.is_true} yes="match" no="non-match" />
                        </Td>
                      </tr>
                    ))}
                  </Table>
                  <p className="mt-1 text-[11px] text-ink-400">
                    Red rows are false merges (emitted, not a match). Amber rows are missed
                    matches (a true match that was retrieved but not emitted).
                  </p>
                </div>
              </div>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
