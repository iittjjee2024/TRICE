/** Currently-selected run, shared across pages. */
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type RunSummary } from "../api/client";

interface Ctx {
  runId: string | null;
  setRunId: (id: string) => void;
  runs: RunSummary[];
  current: RunSummary | null;
  loading: boolean;
  error: unknown;
}

const RunCtx = createContext<Ctx>({
  runId: null,
  setRunId: () => {},
  runs: [],
  current: null,
  loading: false,
  error: null,
});

const KEY = "trice.runId";

export function RunProvider({ children }: { children: ReactNode }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["runs"],
    queryFn: api.runs,
    refetchInterval: 20_000,
  });
  const runs = data?.runs ?? [];
  const [runId, setRunIdState] = useState<string | null>(
    () => localStorage.getItem(KEY),
  );

  useEffect(() => {
    if (!runs.length) return;
    const valid = runs.some((r) => r.run_id === runId);
    if (!valid) {
      // prefer the best-scoring run with validation arrays, else the newest
      const scored = [...runs]
        .filter((r) => r.has_val_pairs && r.headline.val_macro_f05 != null)
        .sort(
          (a, b) =>
            (b.headline.val_macro_f05 ?? 0) - (a.headline.val_macro_f05 ?? 0),
        );
      const pick = scored[0]?.run_id ?? runs[0].run_id;
      setRunIdState(pick);
      localStorage.setItem(KEY, pick);
    }
  }, [runs, runId]);

  const value = useMemo<Ctx>(
    () => ({
      runId,
      setRunId: (id: string) => {
        setRunIdState(id);
        localStorage.setItem(KEY, id);
      },
      runs,
      current: runs.find((r) => r.run_id === runId) ?? null,
      loading: isLoading,
      error,
    }),
    [runId, runs, isLoading, error],
  );

  return <RunCtx.Provider value={value}>{children}</RunCtx.Provider>;
}

export const useRun = () => useContext(RunCtx);
