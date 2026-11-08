import { Fragment, useEffect, useState } from "react";
import { api, Dataset, Job } from "../api";

type Ev = {
  id: number;
  status: string;
  error: string | null;
  label: string;
  base_model: string;
  dataset_id: number;
  dataset_name: string;
  contaminated: boolean;
  result: null | {
    rows: number;
    loss: number;
    perplexity: number;
    f1: number;
    exact_match: number;
    avg_gen_seconds: number;
    samples: { prompt: string; reference: string; prediction: string }[];
  };
};

const pct = (v: number) => (v * 100).toFixed(1) + " %";

export default function Evaluation({ pid }: { pid: number }) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [trained, setTrained] = useState<Job[]>([]);
  const [evals, setEvals] = useState<Ev[]>([]);
  const [dsId, setDsId] = useState(0);
  const [picked, setPicked] = useState<Set<string>>(new Set(["base"]));
  const [baseModel, setBaseModel] = useState("Qwen/Qwen2.5-0.5B-Instruct");
  const [maxRows, setMaxRows] = useState(50);
  const [msg, setMsg] = useState("");
  const [open, setOpen] = useState<number | null>(null);

  const refresh = () => api(`/projects/${pid}/evals`).then(setEvals).catch(() => {});

  useEffect(() => {
    api(`/projects/${pid}/datasets?kind=text`)
      .then((d: Dataset[]) => {
        setDatasets(d);
        setDsId((cur) => cur || d[0]?.id || 0);
      })
      .catch(() => {});
    api(`/projects/${pid}/jobs`)
      .then((j: Job[]) => setTrained(j.filter((x) => x.engine === "llm" && x.kind === "train" && x.status === "completed")))
      .catch(() => {});
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [pid]);

  const toggle = (k: string) =>
    setPicked((s) => {
      const n = new Set(s);
      n.has(k) ? n.delete(k) : n.add(k);
      return n;
    });

  const run = async () => {
    setMsg("");
    try {
      const warns: string[] = [];
      for (const k of picked) {
        const r = await api(`/projects/${pid}/eval`, {
          body: { dataset_id: dsId, job_id: k === "base" ? null : Number(k), base_model: baseModel, max_rows: maxRows },
        });
        if (r.contaminated) warns.push(`#${k}`);
      }
      if (warns.length) setMsg(`Aviso: ${warns.join(", ")} se entrenó con este mismo dataset; el resultado estará inflado.`);
      refresh();
    } catch (e: any) {
      setMsg(e.message);
    }
  };

  // Mejor valor por columna dentro del mismo dataset, para resaltarlo.
  const done = evals.filter((e) => e.result && e.dataset_id === dsId);
  const best = (k: "loss" | "perplexity" | "f1" | "exact_match") => {
    const vals = done.map((e) => e.result![k]);
    return vals.length < 2 ? null : k === "f1" || k === "exact_match" ? Math.max(...vals) : Math.min(...vals);
  };

  const cell = (e: Ev, k: "loss" | "perplexity" | "f1" | "exact_match", fmt: (v: number) => string) => {
    const v = e.result?.[k];
    if (v === undefined) return <td>—</td>;
    return <td className={best(k) === v && e.dataset_id === dsId ? "best" : ""}>{fmt(v)}</td>;
  };

  return (
    <>
      <section className="card wide">
        <h3>Nueva evaluación</h3>
        <p className="muted small">
          Mide la pérdida y perplejidad sobre la respuesta esperada, y compara la respuesta generada con la de
          referencia (F1 de palabras y coincidencia exacta). Para que sea fiable, usa un dataset que{" "}
          <b>no</b> se haya empleado en el entrenamiento.
        </p>
        <div className="grid">
          <label>
            Dataset de evaluación
            <select value={dsId} onChange={(e) => setDsId(Number(e.target.value))}>
              {datasets.length === 0 && <option value={0}>— importa un dataset primero —</option>}
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.rows})
                </option>
              ))}
            </select>
          </label>
          <label>
            Máx. de filas
            <input type="number" value={maxRows} onChange={(e) => setMaxRows(Number(e.target.value))} />
          </label>
          <label className="check">
            <input type="checkbox" checked={picked.has("base")} onChange={() => toggle("base")} />
            Modelo base
            <input value={baseModel} onChange={(e) => setBaseModel(e.target.value)} style={{ flex: 1 }} />
          </label>
          {trained.map((j) => (
            <label key={j.id} className="check">
              <input type="checkbox" checked={picked.has(String(j.id))} onChange={() => toggle(String(j.id))} />
              Entrenamiento #{j.id} · {j.config.base_model}
            </label>
          ))}
        </div>
        <button className="primary" disabled={!dsId || picked.size === 0} onClick={run}>
          Evaluar {picked.size} modelo{picked.size === 1 ? "" : "s"}
        </button>
        {msg && <p className="msg">{msg}</p>}
      </section>

      <section className="card wide">
        <h3>Comparación</h3>
        {evals.length === 0 && <p className="muted">Aún no hay evaluaciones.</p>}
        {evals.length > 0 && (
          <table className="tbl">
            <thead>
              <tr>
                <th>Modelo</th>
                <th>Dataset</th>
                <th>Estado</th>
                <th>Loss ↓</th>
                <th>Perplejidad ↓</th>
                <th>F1 ↑</th>
                <th>Exacta ↑</th>
                <th>s/resp.</th>
              </tr>
            </thead>
            <tbody>
              {evals.map((e) => (
                <Fragment key={e.id}>
                  <tr onClick={() => setOpen(open === e.id ? null : e.id)} style={{ cursor: "pointer" }}>
                    <td>
                      {e.label}
                      {e.contaminated && <span className="tag failed" title="Evaluado con sus propios datos de entrenamiento"> sesgado</span>}
                    </td>
                    <td>{e.dataset_name}</td>
                    <td>
                      <span className={"tag " + e.status}>{e.status}</span>
                      {e.error && <div className="msg small">{e.error}</div>}
                    </td>
                    {cell(e, "loss", (v) => v.toFixed(3))}
                    {cell(e, "perplexity", (v) => v.toFixed(2))}
                    {cell(e, "f1", pct)}
                    {cell(e, "exact_match", pct)}
                    <td>{e.result ? e.result.avg_gen_seconds.toFixed(2) : "—"}</td>
                  </tr>
                  {open === e.id && e.result && (
                    <tr>
                      <td colSpan={8}>
                        {e.result.samples.map((s, i) => (
                          <div key={i} className="sample">
                            <div className="muted small">{s.prompt}</div>
                            <div>
                              <b>Esperado:</b> {s.reference}
                            </div>
                            <div>
                              <b>Generado:</b> {s.prediction || "(vacío)"}
                            </div>
                          </div>
                        ))}
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        )}
        <p className="muted small">En verde, el mejor resultado entre los modelos evaluados con el dataset seleccionado. Pulsa una fila para ver ejemplos.</p>
      </section>
    </>
  );
}
