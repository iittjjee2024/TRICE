import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "../api/client";
import { Card, ErrorBox, Spinner } from "../components/ui";

export default function DocsPage() {
  const list = useQuery({ queryKey: ["docs"], queryFn: api.docsList });
  const [name, setName] = useState<string | null>(null);

  useEffect(() => {
    if (!name && list.data?.docs.length) setName(list.data.docs[0].name);
  }, [list.data, name]);

  const doc = useQuery({
    queryKey: ["doc", name],
    queryFn: () => api.doc(name!),
    enabled: !!name,
  });

  return (
    <div className="grid gap-5 lg:grid-cols-[260px_1fr]">
      <Card title="Documents">
        {list.isLoading ? (
          <Spinner />
        ) : list.error ? (
          <ErrorBox error={list.error} />
        ) : (
          <nav>
            <ul className="space-y-1">
              {list.data!.docs.map((d) => (
                <li key={d.name}>
                  <button
                    onClick={() => setName(d.name)}
                    aria-current={name === d.name ? "true" : undefined}
                    className={`w-full rounded-md px-2.5 py-1.5 text-left text-[12.5px] ${
                      name === d.name
                        ? "bg-accent/15 text-accent"
                        : "text-ink-300 hover:bg-ink-800 hover:text-white"
                    }`}
                  >
                    <span className="block font-medium">{d.title}</span>
                    <span className="mono block text-[10.5px] text-ink-400">
                      {d.name} · {(d.bytes / 1024).toFixed(0)} KB
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </nav>
        )}
      </Card>

      <Card title={doc.data?.name ?? "Document"}>
        {doc.isLoading ? (
          <Spinner label="Loading document" />
        ) : doc.error ? (
          <ErrorBox error={doc.error} />
        ) : (
          <article className="prose-trice max-w-none">
            <Markdown remarkPlugins={[remarkGfm]}>{doc.data?.markdown ?? ""}</Markdown>
          </article>
        )}
      </Card>
    </div>
  );
}
