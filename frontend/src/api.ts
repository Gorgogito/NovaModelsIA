export const API = "http://127.0.0.1:8765";

export async function api<T = any>(path: string, opts?: { method?: string; body?: unknown }): Promise<T> {
  const r = await fetch(API + path, {
    method: opts?.method ?? (opts?.body ? "POST" : "GET"),
    headers: opts?.body ? { "Content-Type": "application/json" } : undefined,
    body: opts?.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = data?.detail;
    throw new Error(
      typeof d === "string" ? d : d?.msg ?? (Array.isArray(d) ? d.map((x: any) => x.msg).join("; ") : `Error ${r.status}`),
    );
  }
  return data;
}

export type Project = { id: number; name: string };
export type Dataset = { id: number; name: string; rows: number; license: string; report: any };
export type Job = {
  id: number;
  kind: string;
  engine: string;
  status: string;
  error: string | null;
  output_dir: string;
  config: Record<string, any>;
  created_at: number;
};
