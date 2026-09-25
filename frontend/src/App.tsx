import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api/client";
import { RunProvider, useRun } from "./state/RunContext";
import Dashboard from "./pages/Dashboard";
import Datasets from "./pages/Datasets";
import Blocking from "./pages/Blocking";
import ModelPage from "./pages/ModelPage";
import Calibration from "./pages/Calibration";
import DecisionTuner from "./pages/DecisionTuner";
import Entities from "./pages/Entities";
import Output from "./pages/Output";
import DocsPage from "./pages/DocsPage";
import { Pill } from "./components/ui";

const NAV = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/datasets", label: "Data" },
  { to: "/blocking", label: "Blocking" },
  { to: "/model", label: "Model" },
  { to: "/calibration", label: "Calibration" },
  { to: "/decision", label: "Decision tuner" },
  { to: "/entities", label: "Entities" },
  { to: "/output", label: "Submission" },
  { to: "/docs", label: "Docs" },
];

function RunPicker() {
  const { runId, setRunId, runs } = useRun();
  if (!runs.length) return <Pill tone="warn">no runs yet</Pill>;
  return (
    <label className="flex items-center gap-2">
      <span className="text-[11px] tracking-wide text-ink-400 uppercase">Run</span>
      <select
        value={runId ?? ""}
        onChange={(e) => setRunId(e.target.value)}
        className="rounded-md border border-ink-600 bg-ink-900 px-2 py-1 text-[12px] text-ink-200 focus:border-accent focus:outline-none"
      >
        {runs.map((r) => (
          <option key={r.run_id} value={r.run_id}>
            {r.run_id}
            {r.headline.val_macro_f05
              ? `  ·  F0.5 ${r.headline.val_macro_f05.toFixed(4)}`
              : ""}
          </option>
        ))}
      </select>
    </label>
  );
}

function Shell() {
  const location = useLocation();
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });

  return (
    <div className="flex min-h-full flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded focus:bg-ink-800 focus:px-3 focus:py-2"
      >
        Skip to content
      </a>

      <header className="sticky top-0 z-30 border-b border-ink-700 bg-ink-900/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1680px] items-center gap-5 px-5 py-2.5">
          <div className="flex items-baseline gap-2.5">
            <span className="text-[15px] font-bold tracking-tight text-white">
              TRICE
            </span>
            <span className="hidden text-[11px] text-ink-400 sm:inline">
              Business Entity Resolution · team Vortex
            </span>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <RunPicker />
            {health.data && (
              <Pill tone={health.data.state.store_built ? "good" : "warn"}>
                {health.data.state.store_built ? "store ready" : "store missing"}
              </Pill>
            )}
          </div>
        </div>
        <nav aria-label="Sections" className="mx-auto max-w-[1680px] px-3">
          <ul className="flex flex-wrap gap-0.5">
            {NAV.map((n) => (
              <li key={n.to}>
                <NavLink
                  to={n.to}
                  end={n.end}
                  className={({ isActive }) =>
                    `-mb-px inline-block border-b-2 px-3 py-2 text-[12.5px] font-medium transition-colors ${
                      isActive
                        ? "border-accent text-white"
                        : "border-transparent text-ink-400 hover:text-ink-200"
                    }`
                  }
                >
                  {n.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </header>

      <main
        id="main"
        className="mx-auto w-full max-w-[1680px] flex-1 px-5 py-5"
        key={location.pathname}
      >
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/blocking" element={<Blocking />} />
          <Route path="/model" element={<ModelPage />} />
          <Route path="/calibration" element={<Calibration />} />
          <Route path="/decision" element={<DecisionTuner />} />
          <Route path="/entities" element={<Entities />} />
          <Route path="/entities/:index" element={<Entities />} />
          <Route path="/output" element={<Output />} />
          <Route path="/docs" element={<DocsPage />} />
          <Route path="/docs/:name" element={<DocsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      <footer className="border-t border-ink-700 px-5 py-3 text-[11px] text-ink-400">
        Local single-user tool bound to loopback · no authentication · reads and writes
        files under the project directory
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <RunProvider>
      <Shell />
    </RunProvider>
  );
}
