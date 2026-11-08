import { Fragment, useEffect, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { api, Dataset, Job } from "../api";
import LossChart from "./LossChart";

const TABS = ["Datos", "Entrenar", "Evaluar", "Probar"] as const;
type Tab = (typeof TABS)[number];
const STATUS: Record<string, string> = {
  queued: "En cola",
  running: "En curso",
  completed: "Completado",
  failed: "Falló",
  stopped: "Detenido",
  interrupted: "Interrumpido",
};
const KIND: Record<string, string> = { train: "Entrenamiento", eval: "Evaluación", export: "Exportación ONNX" };

export default function Vision({ pid }: { pid: number }) {
  const [tab, setTab] = useState<Tab>("Datos");
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [env, setEnv] = useState<{ ready: boolean; state: string } | null>(null);

  const refresh = () => {
    api(`/projects/${pid}/datasets?kind=images`).then(setDatasets).catch(() => {});
    api(`/projects/${pid}/jobs`)
      .then((j: Job[]) => setJobs(j.filter((x) => x.engine === "vision")))
      .catch(() => {});
    api("/engines/vision/env").then(setEnv).catch(() => {});
  };
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [pid]);

  const trained = jobs.filter((j) => j.kind === "train" && j.status === "completed");

  return (
    <>
      <div className="tabs">
        {TABS.map((t) => (
          <button key={t} className={t === tab ? "tab active" : "tab"} onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </div>
      {env && !env.ready && (
        <section className="card warn">
          <h3>Motor de visión no instalado</h3>
          <p className="muted small">Instala PyTorch (CUDA), torchvision, timm y ONNX Runtime en un entorno aislado. Descarga varios GB.</p>
          <button
            className="primary"
            disabled={env.state === "installing"}
            onClick={() => api("/engines/vision/env/install", { method: "POST" }).then(refresh)}
          >
            {env.state === "installing" ? "Instalando… (puede tardar)" : "Instalar motor de visión"}
          </button>
          {env.state.startsWith("error") && <p className="msg">{env.state}</p>}
        </section>
      )}
      {tab === "Datos" && <Data pid={pid} datasets={datasets} refresh={refresh} />}
      {tab === "Entrenar" && <Train pid={pid} datasets={datasets} jobs={jobs} ready={!!env?.ready} refresh={refresh} />}
      {tab === "Evaluar" && <Evaluate pid={pid} datasets={datasets} trained={trained} />}
      {tab === "Probar" && <Try trained={trained} />}
    </>
  );
}

function Data({ pid, datasets, refresh }: { pid: number; datasets: Dataset[]; refresh: () => void }) {
  const [dir, setDir] = useState("");
  const [name, setName] = useState("");
  const [license, setLicense] = useState("");
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);

  const pick = async () => {
    try {
      const f = await open({ directory: true });
      if (typeof f === "string") {
        setDir(f);
        if (!name) setName(f.split(/[\\/]/).filter(Boolean).pop() ?? "");
      }
    } catch {
      setMsg("El selector de carpetas solo funciona en la app de escritorio; escribe la ruta.");
    }
  };
  const doImport = async () => {
    setMsg("");
    setBusy(true);
    try {
      const r = await api(`/projects/${pid}/datasets/images`, { body: { name, source_dir: dir, license } });
      setMsg(`Importado: ${r.rows} imágenes en ${Object.keys(r.report.classes).length} clases (${r.report.invalid} inválidas omitidas).`);
      setDir("");
      setName("");
      refresh();
    } catch (e: any) {
      setMsg(e.message);
    }
    setBusy(false);
  };

  return (
    <>
      <section className="card wide">
        <h3>Importar imágenes (una carpeta por clase)</h3>
        <p className="muted small">
          Estructura esperada: <code>carpeta/gato/*.jpg</code>, <code>carpeta/perro/*.jpg</code>… Se valida cada imagen y se guarda una copia en el proyecto.
        </p>
        <div className="row">
          <input value={dir} onChange={(e) => setDir(e.target.value)} placeholder="Carpeta con subcarpetas por clase" />
          <button onClick={pick}>Examinar…</button>
        </div>
        <div className="row">
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Nombre" />
          <input value={license} onChange={(e) => setLicense(e.target.value)} placeholder="Licencia del dataset" />
        </div>
        <button className="primary" disabled={!dir || !name || busy} onClick={doImport}>
          {busy ? "Validando…" : "Importar y validar"}
        </button>
        {msg && <p className="msg">{msg}</p>}
      </section>
      <section className="card wide">
        <h3>Datasets de imágenes</h3>
        {datasets.length === 0 && <p className="muted">Aún no hay datasets de imágenes.</p>}
        {datasets.map((d) => (
          <div key={d.id} className="item" style={{ flexDirection: "column", alignItems: "stretch" }}>
            <div>
              <b>{d.name}</b> · {d.rows} imágenes{d.license && <> · licencia: {d.license}</>}
            </div>
            <div className="small muted">
              {Object.entries(d.report?.classes ?? {}).map(([c, n]) => `${c}: ${n}`).join(" · ")}
            </div>
            {(d.report?.warnings ?? []).map((w: string) => (
              <div key={w} className="msg small">
                ⚠ {w}
              </div>
            ))}
          </div>
        ))}
      </section>
    </>
  );
}

function Train({ pid, datasets, jobs, ready, refresh }: { pid: number; datasets: Dataset[]; jobs: Job[]; ready: boolean; refresh: () => void }) {
  const [models, setModels] = useState<any[]>([]);
  const [form, setForm] = useState<any>({
    dataset_id: 0, model_name: "resnet18.a1_in1k", epochs: 10, batch_size: 32, learning_rate: 0.001,
    img_size: 160, val_split: 0.2, freeze_backbone: false, augment: true,
  });
  const [license, setLicense] = useState("");
  const [sel, setSel] = useState<number | null>(null);
  const [metrics, setMetrics] = useState<any[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [msg, setMsg] = useState("");
  const cursor = useRef({ m: 0, l: 0 });

  useEffect(() => {
    api("/vision/models").then(setModels).catch(() => {});
  }, []);
  useEffect(() => {
    if (!form.dataset_id && datasets[0]) setForm((f: any) => ({ ...f, dataset_id: datasets[0].id }));
  }, [datasets]);

  useEffect(() => {
    setMetrics([]);
    setLogs([]);
    cursor.current = { m: 0, l: 0 };
    if (sel == null) return;
    const poll = async () => {
      const m = await api(`/jobs/${sel}/metrics?after=${cursor.current.m}`);
      const l = await api(`/jobs/${sel}/logs?after=${cursor.current.l}`);
      if (m.length) {
        cursor.current.m = m[m.length - 1].id;
        setMetrics((x) => [...x, ...m.filter((r: any) => r.type === "metric")]);
      }
      if (l.length) {
        cursor.current.l = l[l.length - 1].id;
        setLogs((x) => [...x, ...l.map((r: any) => r.line)].slice(-200));
      }
    };
    poll().catch(() => {});
    const id = setInterval(() => poll().catch(() => {}), 1500);
    return () => clearInterval(id);
  }, [sel]);

  const set = (k: string, v: any) => setForm((f: any) => ({ ...f, [k]: v }));
  const num = (k: string, label: string, step?: number) => (
    <label>
      {label}
      <input type="number" step={step} value={form[k]} onChange={(e) => set(k, Number(e.target.value))} />
    </label>
  );
  const checkLicense = async () => {
    try {
      const r = await api(`/models/license?repo_id=${encodeURIComponent("timm/" + form.model_name)}`);
      setLicense(r.license);
    } catch (e: any) {
      setLicense(`no se pudo consultar (${e.message})`);
    }
  };
  const start = async (resume?: number) => {
    setMsg("");
    try {
      const r = await api(`/projects/${pid}/vision/train`, { body: { ...form, resume_job_id: resume ?? null } });
      setSel(r.job_id);
      refresh();
    } catch (e: any) {
      setMsg(e.message);
    }
  };
  const exportOnnx = async (id: number) => {
    try {
      const r = await api(`/projects/${pid}/vision/export`, { body: { job_id: id } });
      setSel(r.job_id);
      refresh();
    } catch (e: any) {
      setMsg(e.message);
    }
  };

  const last = metrics[metrics.length - 1];
  const selJob = jobs.find((j) => j.id === sel);

  return (
    <>
      <section className="card wide">
        <h3>Nuevo entrenamiento (fine-tuning)</h3>
        <div className="grid">
          <label>
            Dataset
            <select value={form.dataset_id} onChange={(e) => set("dataset_id", Number(e.target.value))}>
              {datasets.length === 0 && <option value={0}>— importa un dataset primero —</option>}
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.rows})
                </option>
              ))}
            </select>
          </label>
          <label>
            Modelo preentrenado
            <div className="row">
              <select
                value={form.model_name}
                onChange={(e) => {
                  set("model_name", e.target.value);
                  setLicense("");
                }}
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} · {m.params_m} M
                  </option>
                ))}
              </select>
              <button onClick={checkLicense}>Licencia</button>
            </div>
            {license && (
              <span className="small muted">
                Licencia: <b>{license}</b>
              </span>
            )}
          </label>
          {num("epochs", "Épocas")}
          {num("batch_size", "Tamaño de lote")}
          {num("learning_rate", "Learning rate", 0.0005)}
          {num("img_size", "Tamaño de imagen (px)")}
          {num("val_split", "Fracción de validación", 0.05)}
          <label className="check">
            <input type="checkbox" checked={form.freeze_backbone} onChange={(e) => set("freeze_backbone", e.target.checked)} />
            Congelar el backbone (solo entrena la última capa)
          </label>
          <label className="check">
            <input type="checkbox" checked={form.augment} onChange={(e) => set("augment", e.target.checked)} />
            Aumento de datos
          </label>
        </div>
        <button className="primary" disabled={!ready || !form.dataset_id} onClick={() => start()}>
          Iniciar entrenamiento
        </button>
        {msg && <p className="msg">{msg}</p>}
      </section>

      <section className="card wide">
        <h3>Trabajos de visión</h3>
        {jobs.length === 0 && <p className="muted">Aún no hay trabajos.</p>}
        {jobs.map((j) => (
          <div key={j.id} className={"item" + (j.id === sel ? " sel" : "")}>
            <div onClick={() => setSel(j.id)} style={{ cursor: "pointer", flex: 1 }}>
              <b>
                #{j.id} {KIND[j.kind] ?? j.kind}
              </b>{" "}
              · <span className={"tag " + j.status}>{STATUS[j.status] ?? j.status}</span>
              {j.kind === "train" && <> · {j.config.model_name}</>}
              {j.error && <div className="msg small">{j.error}</div>}
            </div>
            {["running", "queued"].includes(j.status) && (
              <button onClick={() => api(`/jobs/${j.id}/stop`, { method: "POST" }).then(refresh)}>Detener</button>
            )}
            {j.kind === "train" && ["stopped", "failed", "interrupted"].includes(j.status) && (
              <button onClick={() => start(j.id)}>Reanudar</button>
            )}
            {j.kind === "train" && j.status === "completed" && <button onClick={() => exportOnnx(j.id)}>Exportar ONNX</button>}
          </div>
        ))}
      </section>

      {sel != null && (
        <section className="card wide">
          <h3>
            Monitoreo #{sel} {selJob && `· ${STATUS[selJob.status] ?? selJob.status}`}
          </h3>
          {last && (
            <div className="stats">
              <span>
                Época{" "}
                <b>
                  {last.step}/{last.max_steps}
                </b>
              </span>
              <span>
                Loss <b>{last.loss?.toFixed(4)}</b>
              </span>
              <span>
                Val loss <b>{last.eval_loss?.toFixed(4)}</b>
              </span>
              <span>
                Precisión val. <b>{(last.val_acc * 100).toFixed(1)} %</b>
              </span>
              {last.vram_mb !== undefined && (
                <span>
                  VRAM pico <b>{last.vram_mb} MB</b>
                </span>
              )}
            </div>
          )}
          <LossChart data={metrics} />
          <pre className="logs">{logs.join("\n") || "Sin registros."}</pre>
        </section>
      )}
    </>
  );
}

function Evaluate({ pid, datasets, trained }: { pid: number; datasets: Dataset[]; trained: Job[] }) {
  const [dsId, setDsId] = useState(0);
  const [jobId, setJobId] = useState(0);
  const [evals, setEvals] = useState<any[]>([]);
  const [msg, setMsg] = useState("");
  const [open_, setOpen] = useState<number | null>(null);

  const refresh = () => api(`/projects/${pid}/evals?engine=vision`).then(setEvals).catch(() => {});
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [pid]);
  useEffect(() => {
    if (!dsId && datasets[0]) setDsId(datasets[0].id);
    if (!jobId && trained[0]) setJobId(trained[0].id);
  }, [datasets, trained]);

  const run = async () => {
    setMsg("");
    try {
      const r = await api(`/projects/${pid}/vision/eval`, { body: { dataset_id: dsId, job_id: jobId } });
      if (r.contaminated) setMsg("Aviso: este modelo se entrenó con ese mismo dataset; la precisión estará inflada.");
      refresh();
    } catch (e: any) {
      setMsg(e.message);
    }
  };

  return (
    <>
      <section className="card wide">
        <h3>Nueva evaluación</h3>
        <p className="muted small">
          Para medir de verdad, usa imágenes que <b>no</b> se hayan usado en el entrenamiento (importa otro dataset con las mismas clases).
        </p>
        <div className="row">
          <select value={jobId} onChange={(e) => setJobId(Number(e.target.value))}>
            {trained.length === 0 && <option value={0}>— no hay modelos entrenados —</option>}
            {trained.map((j) => (
              <option key={j.id} value={j.id}>
                Entrenamiento #{j.id} · {j.config.model_name}
              </option>
            ))}
          </select>
          <select value={dsId} onChange={(e) => setDsId(Number(e.target.value))}>
            {datasets.length === 0 && <option value={0}>— sin datasets —</option>}
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} ({d.rows})
              </option>
            ))}
          </select>
          <button className="primary" disabled={!dsId || !jobId} onClick={run}>
            Evaluar
          </button>
        </div>
        {msg && <p className="msg">{msg}</p>}
      </section>

      <section className="card wide">
        <h3>Resultados</h3>
        {evals.length === 0 && <p className="muted">Aún no hay evaluaciones.</p>}
        {evals.length > 0 && (
          <table className="tbl">
            <thead>
              <tr>
                <th>Modelo</th>
                <th>Dataset</th>
                <th>Estado</th>
                <th>Precisión ↑</th>
                <th>F1 macro ↑</th>
                <th>ms/img</th>
              </tr>
            </thead>
            <tbody>
              {evals.map((e) => (
                <Fragment key={e.id}>
                  <tr style={{ cursor: "pointer" }} onClick={() => setOpen(open_ === e.id ? null : e.id)}>
                    <td>
                      {e.label}
                      {e.contaminated && <span className="tag failed"> sesgado</span>}
                    </td>
                    <td>{e.dataset_name}</td>
                    <td>
                      <span className={"tag " + e.status}>{STATUS[e.status] ?? e.status}</span>
                      {e.error && <div className="msg small">{e.error}</div>}
                    </td>
                    <td>{e.result ? (e.result.accuracy * 100).toFixed(1) + " %" : "—"}</td>
                    <td>{e.result ? (e.result.macro_f1 * 100).toFixed(1) + " %" : "—"}</td>
                    <td>{e.result ? e.result.ms_per_image : "—"}</td>
                  </tr>
                  {open_ === e.id && e.result && (
                    <tr>
                      <td colSpan={6}>
                        <Confusion r={e.result} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        )}
        <p className="muted small">Pulsa una fila para ver la matriz de confusión y las métricas por clase.</p>
      </section>
    </>
  );
}

function Confusion({ r }: { r: any }) {
  const max = Math.max(1, ...r.confusion.flat());
  return (
    <div>
      <div className="small muted">Matriz de confusión (filas: real · columnas: predicho) · {r.rows} imágenes</div>
      <table className="tbl cm">
        <thead>
          <tr>
            <th></th>
            {r.classes.map((c: string) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {r.confusion.map((row: number[], i: number) => (
            <tr key={i}>
              <th>{r.classes[i]}</th>
              {row.map((v, j) => (
                <td key={j} style={{ background: `rgba(108,140,255,${(v / max) * (i === j ? 0.8 : 0.5)})`, textAlign: "center" }}>
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <table className="tbl">
        <thead>
          <tr>
            <th>Clase</th>
            <th>Imágenes</th>
            <th>Precisión</th>
            <th>Recall</th>
            <th>F1</th>
          </tr>
        </thead>
        <tbody>
          {r.per_class.map((p: any) => (
            <tr key={p.class}>
              <td>{p.class}</td>
              <td>{p.support}</td>
              <td>{(p.precision * 100).toFixed(1)} %</td>
              <td>{(p.recall * 100).toFixed(1)} %</td>
              <td>{(p.f1 * 100).toFixed(1)} %</td>
            </tr>
          ))}
        </tbody>
      </table>
      {r.errors.length > 0 && (
        <div className="small muted">
          Errores: {r.errors.map((e: any) => `${e.file} (real ${e.true} → ${e.pred})`).join(" · ")}
        </div>
      )}
    </div>
  );
}

function Try({ trained }: { trained: Job[] }) {
  const [jobId, setJobId] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [img, setImg] = useState("");
  const [top, setTop] = useState<{ class: string; prob: number }[]>([]);

  useEffect(() => {
    if (!jobId && trained[0]) setJobId(trained[0].id);
  }, [trained]);

  const load = async () => {
    setBusy(true);
    setErr("");
    try {
      await api("/vision/load", { body: { job_id: jobId } });
      setLoaded(true);
      setTop([]);
    } catch (e: any) {
      setErr(e.message);
    }
    setBusy(false);
  };

  const onFile = (f?: File) => {
    if (!f) return;
    const rd = new FileReader();
    rd.onload = async () => {
      const url = String(rd.result);
      setImg(url);
      setBusy(true);
      setErr("");
      try {
        const r = await api("/vision/predict", { body: { image_b64: url.split(",")[1] } });
        setTop(r.top);
      } catch (e: any) {
        setErr(e.message);
      }
      setBusy(false);
    };
    rd.readAsDataURL(f);
  };

  return (
    <section className="card wide">
      <h3>Probar un clasificador</h3>
      <div className="row">
        <select value={jobId} onChange={(e) => setJobId(Number(e.target.value))}>
          {trained.length === 0 && <option value={0}>— no hay modelos entrenados —</option>}
          {trained.map((j) => (
            <option key={j.id} value={j.id}>
              Entrenamiento #{j.id} · {j.config.model_name}
            </option>
          ))}
        </select>
        <button className="primary" disabled={!jobId || busy} onClick={load}>
          {loaded ? "Recargar" : "Cargar"}
        </button>
        {loaded && (
          <button onClick={() => api("/vision/unload", { method: "POST" }).then(() => setLoaded(false))}>Descargar</button>
        )}
      </div>
      <input type="file" accept="image/*" disabled={!loaded || busy} onChange={(e) => onFile(e.target.files?.[0])} />
      {err && <p className="msg">{err}</p>}
      <div className="predict">
        {img && <img src={img} alt="Imagen a clasificar" />}
        <div style={{ flex: 1 }}>
          {top.map((t) => (
            <div key={t.class} className="meter">
              <div className="meter-head">
                <span>{t.class}</span>
                <span>{(t.prob * 100).toFixed(1)} %</span>
              </div>
              <div className="bar">
                <div className="fill" style={{ width: `${t.prob * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
