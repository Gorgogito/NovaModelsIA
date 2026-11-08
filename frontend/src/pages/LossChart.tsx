type P = { step: number; loss?: number; eval_loss?: number };

/** `smooth` > 0 dibuja además la media móvil de esa ventana (útil cuando la pérdida por paso es muy ruidosa, como en difusión). */
export default function LossChart({ data, smooth = 0 }: { data: P[]; smooth?: number }) {
  const W = 600,
    H = 200,
    pad = 28;
  const tr = data.filter((d) => d.loss !== undefined);
  const ev = data.filter((d) => d.eval_loss !== undefined);
  if (tr.length < 2) return <p className="muted">Esperando métricas…</p>;
  const avg =
    smooth > 1
      ? tr.map((d, i) => {
          const win = tr.slice(Math.max(0, i - smooth + 1), i + 1);
          return { step: d.step, loss: win.reduce((a, b) => a + b.loss!, 0) / win.length };
        })
      : [];
  const all = [...tr.map((d) => d.loss!), ...ev.map((d) => d.eval_loss!)];
  const maxS = Math.max(...data.map((d) => d.step));
  // Con media móvil, los picos aislados no deben aplastar la escala: se acota al percentil 95 de los valores crudos.
  const sorted = [...all].sort((a, b) => a - b);
  const maxY = smooth > 1 ? sorted[Math.floor(sorted.length * 0.95)] : Math.max(...all);
  const minY = Math.min(...all);
  const x = (s: number) => pad + (s / (maxS || 1)) * (W - pad * 2);
  const y = (v: number) => H - pad - (Math.min(v, maxY) - minY) / (maxY - minY || 1) * (H - pad * 2);
  const line = (pts: P[], k: "loss" | "eval_loss") => pts.map((d) => `${x(d.step)},${y(d[k]!)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Curva de pérdida">
      <text x={4} y={pad} className="axis">{maxY.toFixed(smooth > 1 ? 3 : 2)}</text>
      <text x={4} y={H - pad} className="axis">{minY.toFixed(smooth > 1 ? 3 : 2)}</text>
      <text x={W - pad} y={H - 8} className="axis" textAnchor="end">paso {maxS}</text>
      <polyline points={line(tr, "loss")} fill="none" stroke="var(--accent)" strokeWidth={avg.length ? 1 : 2} opacity={avg.length ? 0.35 : 1} />
      {avg.length > 0 && <polyline points={line(avg, "loss")} fill="none" stroke="var(--accent)" strokeWidth="2.5" />}
      {ev.length > 0 && (
        <polyline points={line(ev, "eval_loss")} fill="none" stroke="#e8a43a" strokeWidth="2" strokeDasharray="4 3" />
      )}
    </svg>
  );
}
