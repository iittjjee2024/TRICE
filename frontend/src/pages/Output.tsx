import { useState } from "react";
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
  btnCls,
  btnPrimaryCls,
  fmt,
  num,
} from "../components/ui";

const CHECK_LABEL: Record<string, string> = {
  one_row_per_s1: "Exactly one row per Source 1 entity",
  all_test_s1_present: "Every test Source 1 entity present",
  no_unknown_s1_rows: "No rows for unknown Source 1 ids",
  no_duplicate_ids_within_list: "No duplicate ids inside a list",
  no_self_matches: "No Source 1 self-matches",
  only_s2_s3_ids: "Only S2-/S3- ids",
  all_ids_exist_in_test: "All emitted ids exist in the test set",
  candidates_cover_all_s1: "candidate_pairs.tsv covers every entity",
  matches_subset_of_candidates: "Matches are a subset of candidates",
};

export default function Output() {
  const { runId } = useRun();
  const [checkIds, setCheckIds] = useState(false);
  const status = useQuery({ queryKey: ["output"], queryFn: api.output });
  const validation = useQuery({
    queryKey: ["validate", checkIds],
    queryFn: () => api.validateOutput(checkIds),
    retry: false,
  });
  const inference = useQuery({
    queryKey: ["inference", runId],
    queryFn: () => api.inference(runId!),
    enabled: !!runId,
    retry: false,
  });

  return (
    <div className="space-y-5">
      <Card
        title="Submission package"
        subtitle="output/matching_results.tsv is the only file scored on the leaderboard"
      >
        <pre className="mono overflow-x-auto rounded-lg border border-ink-700 bg-ink-900 p-3 text-[11.5px] leading-relaxed text-ink-300">
{`Vortex_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (upload this to the Portal)
│   └── candidate_pairs.tsv         # final candidate set the model scored
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # trice package (no web dependencies)
│       ├── README.md               # end-to-end reproduction
│       └── requirements.txt        # pinned
└── Documentation_template.md       # filled-in methodology`}
        </pre>
      </Card>

      {status.isLoading ? (
        <Spinner label="Checking output files" />
      ) : status.error ? (
        <ErrorBox error={status.error} />
      ) : (
        <div className="grid gap-5 lg:grid-cols-2">
          {status.data!.files.map((f) => (
            <Card
              key={f.name}
              title={f.name}
              right={
                f.exists ? (
                  <a className={btnCls} href={`/api/output/${f.name}`} download>
                    download
                  </a>
                ) : (
                  <Pill tone="warn">not written</Pill>
                )
              }
            >
              {!f.exists ? (
                <Empty>
                  Run <code className="mono">python scripts/06_infer.py --run-id {runId ?? "&lt;run&gt;"}</code>
                </Empty>
              ) : (
                <>
                  <div className="mb-3 grid grid-cols-2 gap-3">
                    <Stat label="Size" value={`${fmt(f.mb, 2)} MB`} />
                    <Stat label="Bytes" value={num(f.bytes)} />
                  </div>
                  <pre className="mono max-h-56 overflow-auto rounded-lg border border-ink-700 bg-ink-900 p-3 text-[11px] leading-relaxed whitespace-pre text-ink-300">
                    {f.head?.join("\n")}
                  </pre>
                </>
              )}
            </Card>
          ))}
        </div>
      )}

      <Card
        title="Format validation"
        subtitle="Mirrors every rule in the problem statement, one for one"
        right={
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-1.5 text-[12px] text-ink-300">
              <input
                type="checkbox"
                checked={checkIds}
                onChange={(e) => setCheckIds(e.target.checked)}
                className="accent-[#4ea1ff]"
              />
              verify id existence (slow)
            </label>
            <button
              className={btnPrimaryCls}
              onClick={() => validation.refetch()}
              disabled={validation.isFetching}
            >
              {validation.isFetching ? "checking…" : "re-validate"}
            </button>
          </div>
        }
      >
        {validation.isLoading ? (
          <Spinner label="Validating" />
        ) : validation.error ? (
          <ErrorBox error={validation.error} />
        ) : (
          <div className="space-y-4">
            <div
              className={`rounded-lg border px-4 py-3 text-[13px] ${
                validation.data!.ok
                  ? "border-good/40 bg-good/10 text-good"
                  : "border-bad/40 bg-bad/10 text-bad"
              }`}
              role="status"
            >
              <strong>
                {validation.data!.ok ? "PASS" : "FAIL"}
              </strong>{" "}
              <span className="text-ink-200">
                {validation.data!.ok
                  ? "files satisfy every checked rule and are safe to submit"
                  : `${validation.data!.issues.length} issue(s) to fix`}
              </span>
            </div>

            {!!validation.data!.issues.length && (
              <ol className="list-decimal space-y-1 pl-5 text-[12.5px] text-bad">
                {validation.data!.issues.map((i, n) => (
                  <li key={n}>{i}</li>
                ))}
              </ol>
            )}
            {!!validation.data!.warnings.length && (
              <ul className="list-disc space-y-1 pl-5 text-[12.5px] text-warn">
                {validation.data!.warnings.map((w, n) => (
                  <li key={n}>{w}</li>
                ))}
              </ul>
            )}

            <Table head={["Rule", "Result"]}>
              {Object.entries(validation.data!.checks).map(([k, v]) => (
                <tr key={k} className="border-b border-ink-800/70">
                  <Td>{CHECK_LABEL[k] ?? k}</Td>
                  <Td>
                    <Flag ok={v} yes="pass" no="fail" />
                  </Td>
                </tr>
              ))}
            </Table>

            {validation.data!.matching && (
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
                <Stat label="Rows" value={num(validation.data!.matching.rows)} />
                <Stat
                  label="Non-empty"
                  value={num(validation.data!.matching.nonempty)}
                />
                <Stat
                  label="Empty (singletons)"
                  value={num(validation.data!.matching.empty)}
                />
                <Stat
                  label="Total ids emitted"
                  value={num(validation.data!.matching.total_ids)}
                />
                <Stat
                  label="Mean ids / non-empty"
                  value={fmt(validation.data!.matching.mean_ids_per_nonempty, 3)}
                />
              </div>
            )}
          </div>
        )}
      </Card>

      {runId && (
        <Card title="Inference record" subtitle={`run ${runId}`}>
          {inference.isLoading ? (
            <Spinner />
          ) : inference.error ? (
            <Empty>
              No inference has been run for this run yet. Execute{" "}
              <code className="mono">python scripts/06_infer.py --run-id {runId}</code>.
            </Empty>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                <Stat
                  label="Entities written"
                  value={num(inference.data!.total_entities)}
                />
                <Stat
                  label="With predictions"
                  value={num(inference.data!.total_predicted_nonempty)}
                />
                <Stat
                  label="Wall time"
                  value={`${fmt(inference.data!.seconds / 60, 1)} min`}
                />
                <Stat
                  label="Validated"
                  value={<Flag ok={!!inference.data!.validation?.ok} yes="pass" no="fail" />}
                />
              </div>
              <Table
                head={[
                  "Country",
                  "Entities",
                  "Index",
                  "Pairs",
                  "Mean cands",
                  "No candidates",
                  "Selected",
                  "Mean |S|",
                  "Empty preds",
                  "Seconds",
                ]}
              >
                {Object.entries(inference.data!.per_country ?? {}).map(
                  ([c, v]: [string, any]) => (
                    <tr key={c} className="border-b border-ink-800/70">
                      <Td className="font-medium text-ink-200">{c}</Td>
                      <Td className="tnum">{num(v.n_entities)}</Td>
                      <Td className="tnum">{num(v.n_index)}</Td>
                      <Td className="tnum">{num(v.n_pairs)}</Td>
                      <Td className="tnum">{fmt(v.mean_candidates, 2)}</Td>
                      <Td className="tnum text-warn">
                        {num(v.entities_without_candidates)}
                      </Td>
                      <Td className="tnum">{num(v.selected_pairs)}</Td>
                      <Td className="tnum">{fmt(v.mean_k, 3)}</Td>
                      <Td className="tnum">{num(v.empty_predictions)}</Td>
                      <Td className="tnum text-ink-400">{fmt(v.seconds, 0)}</Td>
                    </tr>
                  ),
                )}
              </Table>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
