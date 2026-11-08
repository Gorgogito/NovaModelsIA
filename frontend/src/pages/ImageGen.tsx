import { useEffect, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { API, api, Dataset, Job } from "../api";
import LossChart from "./LossChart";

const TABS = ["Datos", "Entrenar", "Generar", "Comparar"] as const;
type Tab = (typeof TABS)[number];
const STATUS: Record<string, string> = {
  queued: "En cola",
  running: "En curso",
  completed: "Completado",
  failed: "Falló",
  stopped: "Detenido",
  interrupted: "Interrumpido",
};
const KIND: Record<string, string> = { train: "Entrenamiento LoRA", eval: "Comparación" };
const fileUrl = (jobId: number, name: string) => `${API}/jobs/${jobId}/file?name=${encodeURIComponent(name)}`;

export default function ImageGen({ pid }: { pid: number }) {
  const [tab, setTab] = useState<Tab>("Datos");
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [env, setEnv] = useState<{ ready: boolean; state: string } | null>(null);

  const refresh = () => {
    api(`/projects/${pid}/datasets?kind=imagegen`).then(setDatasets).catch(() => {});
    api(`/projects/${pid}/jobs`)
      .then((j: Job[]) => setJobs(j.filter((x) => x.engine === "image")))
      .catch(() => {});
    api("/engines/image/env").then(setEnv).catch(() => {});
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
          <h3>Motor de imagen no instalado</h3>
          <p className="muted small">Instala PyTorch (CUDA), Diffusers, PEFT y bitsandbytes en un entorno aislado. Descarga varios GB.</p>
          <button
            className="primary"
            disabled={env.state === "installing"}
            onClick={() => api("/engines/image/env/install", { method: "POST" }).then(refresh)}
          >
            {env.state === "installing" ? "Instalando… (puede tardar)" : "Instalar motor de imagen"}
          </button>
          {env.state.startsWith("error") && <p className="msg">{env.state}</p>}
        </section>
      )}
      <p className="muted small">
        Modelo base: Stable Diffusion 1.5 (licencia CreativeML OpenRAIL-M, con restricciones de uso). La primera vez descarga unos 2 GB. El comprobador de
        seguridad del modelo no se usa: las imágenes generadas no se filtran.
      </p>
      {tab === "Datos" && <Data pid={pid} datasets={datasets} refresh={refresh} />}
      {tab === "Entrenar" && <Train pid={pid} datasets={datasets} jobs={jobs} ready={!!env?.ready} refresh={refresh} />}
      {tab === "Generar" && <Generate trained={trained} />}
      {tab === "Comparar" && <Compare pid={pid} trained={trained} ready={!!env?.ready} />}
    </>
  );
}

function Data({ pid, datasets, refresh }: { pid: number; datasets: Dataset[]; refresh: () => void }) {
  const [dir, setDir] = useState("");
  const [name, setName] = useState("");
  const [caption, setCaption] = useState("");
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
      const r = await api(`/projects/${pid}/datasets/imagecaps`, { body: { name, source_dir: dir, default_caption: caption, license } });
      setMsg(`Importado: ${r.rows} imágenes (${r.report.invalid} descartadas).`);
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
        <h3>Importar imágenes con descripción</h3>
        <p className="muted small">
          Carpeta con imágenes (<code>.png .jpg .webp</code>). Cada una necesita una descripción: en un <code>metadata.csv</code> (<code>file,text</code>), en un{" "}
          <code>.txt</code> con el mismo nombre, o una descripción por defecto para todas (útil para un estilo). Incluye una palabra clave propia (p. ej.{" "}
          <code>novastyle</code>) para poder invocar lo aprendido.
        </p>
        <div className="row">
          <input value={dir} onChange={(e) => setDir(e.target.value)} placeholder="Carpeta con las imágenes" />
          <button onClick={pick}>Examinar…</button>
        </div>
        <div className="row">
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Nombre" />
          <input value={caption} onChange={(e) => setCaption(e.target.value)} placeholder="Descripción por defecto (opcional)" />
        </div>
        <div className="row">
          <input value={license} onChange={(e) => setLicense(e.target.value)} placeholder="Licencia de las imágenes (importante: ¿puedes entrenar con ellas?)" />
        </div>
        <button className="primary" disabled={!dir || !name || busy} onClick={doImport}>
          {busy ? "Validando…" : "Importar y validar"}
        </button>
        {msg && <p className="msg">{msg}</p>}
      </section>
      <section className="card wide">
        <h3>Datasets de imagen</h3>
        {datasets.length === 0 && <p className="muted">Aún no hay datasets.</p>}
        {datasets.map((d) => (
          <div key={d.id} className="item" style={{ flexDirection: "column", alignItems: "stretch" }}>
            <div>
              <b>{d.name}</b> · {d.rows} imágenes · lado mínimo {d.report?.min_side}px{d.license && <> · licencia: {d.license}</>}
            </div>
            {(d.report?.sample_captions ?? []).map((c: string) => (
              <div key={c} className="small muted">
                “{c}”
              </div>
            ))}
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
  const [form, setForm] = useState<any>({
    dataset_id: 0, resolution: 512, max_steps: 300, batch_size: 1, grad_accum: 4, learning_rate: 0.0001, lora_rank: 8,
    lora_alpha: 8, save_steps: 100, sample_steps: 100, validation_prompt: "", gradient_checkpointing: true,
    use_8bit_adam: true, random_flip: true, center_crop: true,
  });
  const [sel, setSel] = useState<number | null>(null);
  const [metrics, setMetrics] = useState<any[]>([]);
  const [samples, setSamples] = useState<any[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [msg, setMsg] = useState("");
  const cursor = useRef({ m: 0, l: 0 });

  useEffect(() => {
    if (!form.dataset_id && datasets[0]) setForm((f: any) => ({ ...f, dataset_id: datasets[0].id }));
  }, [datasets]);

  useEffect(() => {
    setMetrics([]);
    setSamples([]);
    setLogs([]);
    cursor.current = { m: 0, l: 0 };
    if (sel == null) return;
    const poll = async () => {
      const m = await api(`/jobs/${sel}/metrics?after=${cursor.current.m}`);
      const l = await api(`/jobs/${sel}/logs?after=${cursor.current.l}`);
      if (m.length) {
        cursor.current.m = m[m.length - 1].id;
        setMetrics((x) => [...x, ...m.filter((r: any) => r.type === "metric")]);
        setSamples((x) => [...x, ...m.filter((r: any) => r.type === "sample")]);
      }
      if (l.length) {
        cursor.current.l = l[l.length - 1].id;
        setLogs((x) => [...x, ...l.map((r: any) => r.line)].slice(-200));
      }
    };
    poll().catch(() => {});
    const id = setInterval(() => poll().catch(() => {}), 2000);
    return () => clearInterval(id);
  }, [sel]);

  const set = (k: string, v: any) => setForm((f: any) => ({ ...f, [k]: v }));
  const num = (k: string, label: string, step?: number) => (
    <label>
      {label}
      <input type="number" step={step} value={form[k]} onChange={(e) => set(k, Number(e.target.value))} />
    </label>
  );
  const start = async (resume?: number) => {
    setMsg("");
    try {
      const r = await api(`/projects/${pid}/image/train`, { body: { ...form, resume_job_id: resume ?? null } });
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
        <h3>Nuevo entrenamiento de LoRA</h3>
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
            Resolución
            <select value={form.resolution} onChange={(e) => set("resolution", Number(e.target.value))}>
              {[384, 448, 512, 576].map((r) => (
                <option key={r} value={r}>
                  {r} px {r === 512 ? "(recomendado)" : ""}
                </option>
              ))}
            </select>
          </label>
          {num("max_steps", "Pasos de entrenamiento")}
          {num("learning_rate", "Learning rate", 0.00005)}
          {num("lora_rank", "Rank del LoRA")}
          {num("lora_alpha", "Alpha del LoRA")}
          {num("grad_accum", "Acumulación de gradiente")}
          {num("sample_steps", "Imagen de muestra cada N pasos (0 = nunca)")}
          <label style={{ gridColumn: "1 / -1" }}>
            Prompt de las muestras (vacío = primera descripción del dataset)
            <input value={form.validation_prompt} onChange={(e) => set("validation_prompt", e.target.value)} />
          </label>
          <label className="check">
            <input type="checkbox" checked={form.gradient_checkpointing} onChange={(e) => set("gradient_checkpointing", e.target.checked)} />
            Gradient checkpointing (menos VRAM)
          </label>
          <label className="check">
            <input type="checkbox" checked={form.use_8bit_adam} onChange={(e) => set("use_8bit_adam", e.target.checked)} />
            AdamW de 8 bits (menos VRAM)
          </label>
        </div>
        <button className="primary" disabled={!ready || !form.dataset_id} onClick={() => start()}>
          Iniciar entrenamiento
        </button>
        {msg && <p className="msg">{msg}</p>}
      </section>

      <section className="card wide">
        <h3>Trabajos de imagen</h3>
        {jobs.length === 0 && <p className="muted">Aún no hay trabajos.</p>}
        {jobs.map((j) => (
          <div key={j.id} className={"item" + (j.id === sel ? " sel" : "")}>
            <div onClick={() => setSel(j.id)} style={{ cursor: "pointer", flex: 1 }}>
              <b>
                #{j.id} {KIND[j.kind] ?? j.kind}
              </b>{" "}
              · <span className={"tag " + j.status}>{STATUS[j.status] ?? j.status}</span>
              {j.error && <div className="msg small">{j.error}</div>}
            </div>
            {["running", "queued"].includes(j.status) && (
              <button onClick={() => api(`/jobs/${j.id}/stop`, { method: "POST" }).then(refresh)}>Detener</button>
            )}
            {j.kind === "train" && ["stopped", "failed", "interrupted"].includes(j.status) && <button onClick={() => start(j.id)}>Reanudar</button>}
          </div>
        ))}
      </section>

      {sel != null && selJob?.kind === "train" && (
        <section className="card wide">
          <h3>
            Monitoreo #{sel} {selJob && `· ${STATUS[selJob.status] ?? selJob.status}`}
          </h3>
          {last && (
            <div className="stats">
              <span>
                Paso{" "}
                <b>
                  {last.step}/{last.max_steps}
                </b>
              </span>
              <span>
                Loss <b>{last.loss?.toFixed(4)}</b>
              </span>
              <span>
                VRAM pico <b>{last.vram_mb} MB</b>
              </span>
            </div>
          )}
          <LossChart data={metrics} smooth={15} />
          {samples.length > 0 && (
            <>
              <h3 style={{ marginTop: 16 }}>Muestras durante el entrenamiento</h3>
              <div className="gallery">
                {samples.map((s) => (
                  <figure key={s.id}>
                    <img src={fileUrl(sel, s.file)} alt={`Muestra del paso ${s.step}`} />
                    <figcaption>paso {s.step}</figcaption>
                  </figure>
                ))}
              </div>
            </>
          )}
          <pre className="logs">{logs.join("\n") || "Sin registros."}</pre>
        </section>
      )}
    </>
  );
}

function Generate({ trained }: { trained: Job[] }) {
  const [choice, setChoice] = useState("base");
  const [scale, setScale] = useState(1);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [p, setP] = useState({ prompt: "", negative_prompt: "", steps: 25, guidance: 7.5, seed: 1, width: 512, height: 512, n: 1 });
  const [images, setImages] = useState<{ seed: number; png_b64: string }[]>([]);
  const [info, setInfo] = useState("");

  const load = async () => {
    setBusy(true);
    setErr("");
    try {
      await api("/image/load", { body: { job_id: choice === "base" ? null : Number(choice), lora_scale: scale } });
      setLoaded(true);
      setImages([]);
    } catch (e: any) {
      setErr(e.message);
    }
    setBusy(false);
  };
  const generate = async () => {
    setBusy(true);
    setErr("");
    try {
      const r = await api("/image/generate", { body: p });
      setImages(r.images);
      setInfo(`${r.images.length} imagen(es) en ${r.seconds} s · VRAM pico ${r.vram_mb} MB`);
    } catch (e: any) {
      setErr(e.message);
    }
    setBusy(false);
  };
  const set = (k: string, v: any) => setP((x) => ({ ...x, [k]: v }));

  return (
    <section className="card wide">
      <h3>Generar imágenes</h3>
      <div className="row">
        <select value={choice} onChange={(e) => setChoice(e.target.value)}>
          <option value="base">Solo el modelo base</option>
          {trained.map((j) => (
            <option key={j.id} value={j.id}>
              LoRA del entrenamiento #{j.id}
            </option>
          ))}
        </select>
        <label className="small muted" style={{ whiteSpace: "nowrap" }}>
          Fuerza del LoRA {scale}
          <input type="range" min="0" max="1.5" step="0.1" value={scale} onChange={(e) => setScale(Number(e.target.value))} />
        </label>
        <button className="primary" disabled={busy} onClick={load}>
          {loaded ? "Recargar" : "Cargar"}
        </button>
        {loaded && <button onClick={() => api("/image/unload", { method: "POST" }).then(() => setLoaded(false))}>Descargar</button>}
      </div>
      <input value={p.prompt} onChange={(e) => set("prompt", e.target.value)} placeholder="Prompt (en inglés funciona mejor con SD 1.5)" style={{ width: "100%", marginBottom: 8 }} />
      <input value={p.negative_prompt} onChange={(e) => set("negative_prompt", e.target.value)} placeholder="Prompt negativo (opcional)" style={{ width: "100%", marginBottom: 8 }} />
      <div className="grid" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
        <label>
          Pasos
          <input type="number" value={p.steps} onChange={(e) => set("steps", Number(e.target.value))} />
        </label>
        <label>
          Guidance
          <input type="number" step="0.5" value={p.guidance} onChange={(e) => set("guidance", Number(e.target.value))} />
        </label>
        <label>
          Semilla
          <input type="number" value={p.seed} onChange={(e) => set("seed", Number(e.target.value))} />
        </label>
        <label>
          Imágenes
          <input type="number" min={1} max={4} value={p.n} onChange={(e) => set("n", Number(e.target.value))} />
        </label>
        <label>
          Ancho
          <select value={p.width} onChange={(e) => set("width", Number(e.target.value))}>
            {[256, 384, 512, 640, 768].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </label>
        <label>
          Alto
          <select value={p.height} onChange={(e) => set("height", Number(e.target.value))}>
            {[256, 384, 512, 640, 768].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </label>
      </div>
      <button className="primary" disabled={!loaded || busy || !p.prompt.trim()} onClick={generate}>
        {busy ? "Generando…" : "Generar"}
      </button>
      {err && <p className="msg">{err}</p>}
      {info && <p className="muted small">{info}</p>}
      <div className="gallery">
        {images.map((im) => (
          <figure key={im.seed}>
            <img src={`data:image/png;base64,${im.png_b64}`} alt={`Imagen generada con la semilla ${im.seed}`} />
            <figcaption>
              semilla {im.seed} ·{" "}
              <a href={`data:image/png;base64,${im.png_b64}`} download={`imagen_${im.seed}.png`}>
                guardar
              </a>
            </figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}

function Compare({ pid, trained, ready }: { pid: number; trained: Job[]; ready: boolean }) {
  const [picked, setPicked] = useState<Set<number>>(new Set());
  const [prompts, setPrompts] = useState("");
  const [seeds, setSeeds] = useState("7, 8");
  const [msg, setMsg] = useState("");
  const [evals, setEvals] = useState<any[]>([]);

  const refresh = () => api(`/projects/${pid}/evals?engine=image`).then(setEvals).catch(() => {});
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [pid]);

  const run = async () => {
    setMsg("");
    try {
      await api(`/projects/${pid}/image/compare`, {
        body: {
          job_ids: [...picked],
          prompts: prompts.split("\n").map((s) => s.trim()).filter(Boolean),
          seeds: seeds.split(",").map((s) => Number(s.trim())).filter((n) => Number.isFinite(n)),
        },
      });
      refresh();
    } catch (e: any) {
      setMsg(e.message);
    }
  };
  const toggle = (id: number) =>
    setPicked((s) => {
      const n = new Set(s);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });

  return (
    <>
      <section className="card wide">
        <h3>Comparar modelos</h3>
        <p className="muted small">
          Genera lo mismo (mismo prompt y misma semilla) con el modelo base y con cada LoRA elegido, para ver qué aporta cada uno. Incluye al menos un prompt
          con combinaciones que no estaban en el entrenamiento.
        </p>
        {trained.length === 0 && <p className="muted">Primero entrena un LoRA.</p>}
        {trained.map((j) => (
          <label key={j.id} className="check" style={{ display: "flex", gap: 8, marginBottom: 6 }}>
            <input type="checkbox" checked={picked.has(j.id)} onChange={() => toggle(j.id)} />
            LoRA del entrenamiento #{j.id}
          </label>
        ))}
        <textarea
          value={prompts}
          onChange={(e) => setPrompts(e.target.value)}
          rows={4}
          placeholder={"Un prompt por línea (máx. 8)"}
          style={{ width: "100%", margin: "8px 0", background: "var(--bg)", color: "var(--text)", border: "1px solid var(--line)", borderRadius: 8, padding: 8 }}
        />
        <div className="row">
          <input value={seeds} onChange={(e) => setSeeds(e.target.value)} placeholder="Semillas separadas por comas (máx. 4)" />
          <button className="primary" disabled={!ready || picked.size === 0 || !prompts.trim()} onClick={run}>
            Comparar
          </button>
        </div>
        {msg && <p className="msg">{msg}</p>}
      </section>

      {evals.map((e) => (
        <section key={e.id} className="card wide">
          <h3>
            #{e.id} {e.label} · <span className={"tag " + e.status}>{STATUS[e.status] ?? e.status}</span>
          </h3>
          {e.error && <p className="msg small">{e.error}</p>}
          {e.result && (
            <>
              <div className="cmp" style={{ gridTemplateColumns: `140px repeat(${e.result.columns.length}, 1fr)` }}>
                <div />
                {e.result.columns.map((c: string) => (
                  <div key={c} className="small muted" style={{ textAlign: "center" }}>
                    {c}
                  </div>
                ))}
                {e.result.rows.map((r: any, i: number) => (
                  <Row key={i} jobId={e.id} r={r} />
                ))}
              </div>
              <p className="muted small">{e.result.seconds} s en total.</p>
            </>
          )}
        </section>
      ))}
    </>
  );
}

function Row({ jobId, r }: { jobId: number; r: any }) {
  return (
    <>
      <div className="small">
        {r.prompt}
        <div className="muted">semilla {r.seed}</div>
      </div>
      {r.images.map((rel: string) => (
        <img key={rel} src={fileUrl(jobId, rel)} alt={`${r.prompt} (semilla ${r.seed})`} style={{ width: "100%", borderRadius: 6 }} />
      ))}
    </>
  );
}
