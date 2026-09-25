/** Shared presentational primitives. Dense, dark, data-first. */
import type { ReactNode } from "react";

export function Card({
  title,
  subtitle,
  right,
  children,
  className = "",
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-xl border border-ink-700 bg-ink-850/70 ${className}`}
    >
      {(title || right) && (
        <header className="flex items-start justify-between gap-4 border-b border-ink-700 px-4 py-3">
          <div>
            {title && (
              <h2 className="text-[13px] font-semibold tracking-wide text-ink-200 uppercase">
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="mt-0.5 text-xs text-ink-400">{subtitle}</p>
            )}
          </div>
          {right}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone = "default",
  big = false,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "default" | "good" | "warn" | "bad" | "accent";
  big?: boolean;
}) {
  const toneCls = {
    default: "text-ink-200",
    good: "text-good",
    warn: "text-warn",
    bad: "text-bad",
    accent: "text-accent",
  }[tone];
  return (
    <div className="rounded-lg border border-ink-700 bg-ink-900/60 px-3 py-2.5">
      <div className="text-[10.5px] font-medium tracking-wider text-ink-400 uppercase">
        {label}
      </div>
      <div
        className={`tnum mt-1 font-semibold ${toneCls} ${big ? "text-2xl" : "text-lg"}`}
      >
        {value}
      </div>
      {hint && <div className="mt-0.5 text-[11px] text-ink-400">{hint}</div>}
    </div>
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center gap-2 px-1 py-6 text-sm text-ink-400"
    >
      <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-ink-600 border-t-accent" />
      {label}…
    </div>
  );
}

export function ErrorBox({ error }: { error: unknown }) {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <div
      role="alert"
      className="rounded-lg border border-bad/40 bg-bad/10 px-4 py-3 text-sm text-bad"
    >
      <strong className="font-semibold">Request failed.</strong>{" "}
      <span className="text-ink-200">{msg}</span>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-ink-700 px-4 py-8 text-center text-sm text-ink-400">
      {children}
    </div>
  );
}

/** Match/no-match and pass/fail indicators never rely on colour alone. */
export function Flag({
  ok,
  yes = "yes",
  no = "no",
}: {
  ok: boolean;
  yes?: string;
  no?: string;
}) {
  return (
    <span className={ok ? "text-good" : "text-bad"}>
      <span aria-hidden="true">{ok ? "✓ " : "✕ "}</span>
      {ok ? yes : no}
    </span>
  );
}

export function Pill({
  children,
  tone = "default",
}: {
  children: ReactNode;
  tone?: "default" | "good" | "warn" | "bad" | "accent";
}) {
  const cls = {
    default: "border-ink-600 text-ink-300",
    good: "border-good/50 text-good",
    warn: "border-warn/50 text-warn",
    bad: "border-bad/50 text-bad",
    accent: "border-accent/50 text-accent",
  }[tone];
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium ${cls}`}
    >
      {children}
    </span>
  );
}

export function Table({
  head,
  children,
  dense = false,
}: {
  head: ReactNode[];
  children: ReactNode;
  dense?: boolean;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-[12.5px]">
        <thead>
          <tr className="border-b border-ink-700 text-left">
            {head.map((h, i) => (
              <th
                key={i}
                scope="col"
                className={`${dense ? "px-2 py-1.5" : "px-3 py-2"} text-[10.5px] font-semibold tracking-wider whitespace-nowrap text-ink-400 uppercase`}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function Td({
  children,
  className = "",
  dense = false,
}: {
  children?: ReactNode;
  className?: string;
  dense?: boolean;
}) {
  return (
    <td className={`${dense ? "px-2 py-1" : "px-3 py-1.5"} ${className}`}>
      {children}
    </td>
  );
}

export function fmt(n: number | null | undefined, digits = 4): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

export function pct(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `${(n * 100).toFixed(digits)}%`;
}

export function num(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toLocaleString();
}

export function Labeled({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] font-medium tracking-wide text-ink-300">
        {label}
      </span>
      {children}
      {hint && <span className="mt-1 block text-[11px] text-ink-400">{hint}</span>}
    </label>
  );
}

export const inputCls =
  "w-full rounded-md border border-ink-600 bg-ink-900 px-2.5 py-1.5 text-[13px] text-ink-200 " +
  "placeholder:text-ink-400 focus:border-accent focus:outline-none";

export const selectCls = inputCls + " pr-8";

export const btnCls =
  "inline-flex items-center gap-1.5 rounded-md border border-ink-600 bg-ink-800 px-3 py-1.5 " +
  "text-[12.5px] font-medium text-ink-200 hover:border-accent/60 hover:text-white " +
  "disabled:cursor-not-allowed disabled:opacity-45";

export const btnPrimaryCls =
  "inline-flex items-center gap-1.5 rounded-md border border-accent/60 bg-accent/15 px-3 py-1.5 " +
  "text-[12.5px] font-semibold text-accent hover:bg-accent/25 " +
  "disabled:cursor-not-allowed disabled:opacity-45";
