import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import {
  Card,
  Empty,
  ErrorBox,
  Flag,
  Labeled,
  Pill,
  Spinner,
  Stat,
  Table,
  Td,
  btnPrimaryCls,
  fmt,
  inputCls,
  num,
  selectCls,
} from "../components/ui";

const EXAMPLES: [string, string][] = [
  ["राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi"],
  ["Halonex dba Marnie Baynes Keystone Bnb Inc", "TX, Leander, 1901 Vista Villas Drive"],
  ["edwardshintzedougherty.com", "541 FOURTEENTH SAINT, NEWPORT, OR"],
  ["-- Holloway Peak Inc Seafood", "105 ELM ST, MORGANTON, NC"],
  ["SCI Ptit Àmicale", "18 RUE JEN ZAY, Dunkerque, Nord"],
  ["M/s Finest (India) World Private Ltd", "<NULL>"],
];

export default function Datasets() {
  const files = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const countries = useQuery({
    queryKey: ["dataset-countries"],
    queryFn: api.datasetCountries,
  });
  const variants = useQuery({ queryKey: ["variants"], queryFn: api.variants, retry: false });

  const [split, setSplit] = useState("test");
  const [source, setSource] = useState(1);
  const [country, setCountry] = useState("");
  const preview = useQuery({
    queryKey: ["preview", split, source, country],
    queryFn: () =>
      api.preview({ split, source, limit: 20, ...(country ? { country } : {}) }),
    retry: false,
  });

  const [name, setName] = useState(EXAMPLES[0][0]);
  const [address, setAddress] = useState(EXAMPLES[0][1]);
  const norm = useMutation({ mutationFn: api.normalize });

  return (
    <div className="space-y-5">
      {files.isLoading ? (
        <Spinner label="Reading dataset" />
      ) : files.error ? (
        <ErrorBox error={files.error} />
      ) : (
        <Card
          title="Files"
          subtitle="Raw challenge TSVs and the normalised Parquet record store"
          right={
            <Pill tone={files.data!.prepared ? "good" : "warn"}>
              {files.data!.prepared ? "store built" : "store incomplete"}
            </Pill>
          }
        >
          <Table
            head={["Split", "Source", "File", "Raw", "Raw MB", "Store", "Store MB", "Rows"]}
          >
            {files.data!.files.map((f: any) => (
              <tr key={f.file} className="border-b border-ink-800/70">
                <Td className="text-ink-400">{f.split}</Td>
                <Td className="tnum">{f.source}</Td>
                <Td className="mono text-ink-200">{f.file}</Td>
                <Td>
                  <Flag ok={f.raw_exists} yes="present" no="missing" />
                </Td>
                <Td className="tnum">{fmt(f.raw_mb, 1)}</Td>
                <Td>
                  <Flag ok={f.store_exists} yes="built" no="—" />
                </Td>
                <Td className="tnum">{fmt(f.store_mb, 1)}</Td>
                <Td className="tnum">{num(f.rows)}</Td>
              </tr>
            ))}
          </Table>
        </Card>
      )}

      {countries.data && (
        <Card
          title="Country mix"
          subtitle="France appears only in test — 15% of the score with zero labelled examples"
        >
          <div className="grid gap-4 lg:grid-cols-2">
            {Object.entries(countries.data.countries ?? {}).map(
              ([split, sources]: [string, any]) => (
                <div key={split}>
                  <h3 className="mb-2 text-[11px] tracking-wider text-ink-400 uppercase">
                    {split}
                  </h3>
                  <Table head={["Source", ...Object.keys(sources.source1 ?? {})]}>
                    {Object.entries(sources).map(([src, counts]: [string, any]) => (
                      <tr key={src} className="border-b border-ink-800/70">
                        <Td className="text-ink-200">{src}</Td>
                        {Object.keys(sources.source1 ?? {}).map((c) => (
                          <Td key={c} className="tnum">
                            {num(counts[c] ?? 0)}
                          </Td>
                        ))}
                      </tr>
                    ))}
                  </Table>
                </div>
              ),
            )}
          </div>
        </Card>
      )}

      <Card
        title="Normalisation playground"
        subtitle="Most of this solution's accuracy comes from this stage, so it is directly pokeable"
      >
        <div className="grid gap-4 lg:grid-cols-[1fr_1.2fr]">
          <div className="space-y-3">
            <Labeled label="Business name">
              <input
                className={inputCls}
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </Labeled>
            <Labeled label="Business address">
              <input
                className={inputCls}
                value={address}
                onChange={(e) => setAddress(e.target.value)}
              />
            </Labeled>
            <button
              className={btnPrimaryCls}
              onClick={() => norm.mutate({ name, address })}
              disabled={norm.isPending}
            >
              {norm.isPending ? "normalising…" : "normalise"}
            </button>
            <div>
              <div className="mb-1.5 text-[11px] tracking-wider text-ink-400 uppercase">
                Real examples from the data
              </div>
              <div className="flex flex-wrap gap-1.5">
                {EXAMPLES.map(([n, a], i) => (
                  <button
                    key={i}
                    className="rounded-full border border-ink-600 px-2 py-0.5 text-[11px] text-ink-300 hover:border-accent/60 hover:text-white"
                    onClick={() => {
                      setName(n);
                      setAddress(a);
                      norm.mutate({ name: n, address: a });
                    }}
                  >
                    {n.length > 30 ? n.slice(0, 30) + "…" : n}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div>
            {norm.error ? (
              <ErrorBox error={norm.error} />
            ) : !norm.data ? (
              <Empty>Pick an example or enter your own, then normalise.</Empty>
            ) : (
              <div className="space-y-3 text-[12.5px]">
                <div className="rounded-lg border border-ink-700 bg-ink-900/60 p-3">
                  <div className="mb-1.5 text-[10.5px] tracking-wider text-ink-400 uppercase">
                    Name
                  </div>
                  <KV k="core tokens" v={norm.data.name.core || "(empty)"} />
                  <KV k="legal suffixes" v={norm.data.name.legal.join(", ") || "—"} />
                  <KV k="consonant skeleton" v={norm.data.name.skeleton} />
                  <KV k="space-free" v={norm.data.name.nospace} />
                  <KV
                    k="flags"
                    v={
                      [
                        norm.data.name.is_domain ? "domainified" : null,
                        norm.data.name.had_dba ? "DBA split" : null,
                      ]
                        .filter(Boolean)
                        .join(", ") || "—"
                    }
                  />
                </div>
                <div className="rounded-lg border border-ink-700 bg-ink-900/60 p-3">
                  <div className="mb-1.5 text-[10.5px] tracking-wider text-ink-400 uppercase">
                    Address
                  </div>
                  <KV
                    k="alpha tokens"
                    v={norm.data.address.alpha_tokens.join(" ") || "(empty)"}
                  />
                  <KV k="numeric tokens" v={norm.data.address.digits.join(", ") || "—"} />
                  <KV k="postal" v={norm.data.address.postal || "—"} />
                  <KV k="house" v={norm.data.address.house || "—"} />
                  <KV k="empty" v={String(norm.data.address.is_empty)} />
                </div>
              </div>
            )}
          </div>
        </div>
      </Card>

      <Card
        title="Record store preview"
        subtitle="Normalised fields as the pipeline sees them"
      >
        <div className="mb-3 grid gap-3 sm:grid-cols-3">
          <Labeled label="Split">
            <select
              className={selectCls}
              value={split}
              onChange={(e) => setSplit(e.target.value)}
            >
              <option value="train">train</option>
              <option value="test">test</option>
            </select>
          </Labeled>
          <Labeled label="Source">
            <select
              className={selectCls}
              value={source}
              onChange={(e) => setSource(Number(e.target.value))}
            >
              <option value={1}>source 1</option>
              <option value={2}>source 2</option>
              <option value={3}>source 3</option>
            </select>
          </Labeled>
          <Labeled label="Country">
            <input
              className={inputCls}
              value={country}
              placeholder="all"
              onChange={(e) => setCountry(e.target.value)}
            />
          </Labeled>
        </div>
        {preview.isLoading ? (
          <Spinner />
        ) : preview.error ? (
          <ErrorBox error={preview.error} />
        ) : (
          <>
            <p className="mb-2 text-[11.5px] text-ink-400">
              {num(preview.data!.total)} records match
            </p>
            <Table
              head={[
                "Entity",
                "Country",
                "name_core",
                "name_skel",
                "legal",
                "addr_alpha",
                "digits",
                "postal",
                "house",
              ]}
              dense
            >
              {preview.data!.rows.map((r: any) => (
                <tr key={r.entity_id} className="border-b border-ink-800/70">
                  <Td dense className="mono text-ink-200">
                    {r.entity_id}
                  </Td>
                  <Td dense className="text-ink-400">
                    {r.country}
                  </Td>
                  <Td dense>{r.name_core}</Td>
                  <Td dense className="mono text-ink-400">
                    {r.name_skel}
                  </Td>
                  <Td dense className="text-ink-400">
                    {r.legal}
                  </Td>
                  <Td dense className="max-w-[320px] truncate">{r.addr_alpha}</Td>
                  <Td dense className="mono text-ink-400">
                    {r.addr_digits}
                  </Td>
                  <Td dense className="tnum">
                    {r.postal}
                  </Td>
                  <Td dense className="tnum">
                    {r.house}
                  </Td>
                </tr>
              ))}
            </Table>
          </>
        )}
      </Card>

      <Card
        title="Mined token aliases"
        subtitle="Learned from ground-truth matched pairs — no external gazetteer is consulted"
      >
        {variants.isLoading ? (
          <Spinner />
        ) : variants.error ? (
          <Empty>
            Not mined yet. Run{" "}
            <code className="mono">python scripts/02_mine_variants.py</code>.
          </Empty>
        ) : (
          <>
            <div className="mb-3 grid grid-cols-2 gap-3 lg:grid-cols-4">
              <Stat
                label="Address aliases"
                value={num(variants.data!.n_address_map)}
              />
              <Stat label="Name aliases" value={num(variants.data!.n_name_map)} />
              <Stat
                label="Groups sampled"
                value={num(variants.data!.meta?.groups_used)}
              />
              <Stat
                label="Min association"
                value={fmt(variants.data!.meta?.min_score, 2)}
              />
            </div>
            <div className="grid gap-4 lg:grid-cols-2">
              {(
                [
                  ["Address", variants.data!.address_pairs],
                  ["Name", variants.data!.name_pairs],
                ] as [string, any[]][]
              ).map(([label, pairs]) => (
                <div key={label}>
                  <h3 className="mb-2 text-[11px] tracking-wider text-ink-400 uppercase">
                    {label}
                  </h3>
                  <Table head={["a", "b", "count", "score"]} dense>
                    {pairs.slice(0, 30).map((p: any, i: number) => (
                      <tr key={i} className="border-b border-ink-800/70">
                        <Td dense className="mono">{p.a}</Td>
                        <Td dense className="mono">{p.b}</Td>
                        <Td dense className="tnum">{num(p.count)}</Td>
                        <Td dense className="tnum text-ink-400">{fmt(p.score, 3)}</Td>
                      </tr>
                    ))}
                  </Table>
                </div>
              ))}
            </div>
          </>
        )}
      </Card>
    </div>
  );
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex gap-2 py-0.5">
      <span className="w-[120px] shrink-0 text-ink-400">{k}</span>
      <span className="mono break-all text-ink-200">{v}</span>
    </div>
  );
}
