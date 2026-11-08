import { Fragment, useEffect, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { api, Dataset, Job } from "../api";
import LossChart from "./LossChart";

const TABS = ["Datos", "Entrenar", "Evaluar", "Transcribir"] as const;
type Tab = (typeof TABS)[number];
const STATUS: Record<string, string> = {
  queued: "En cola",
  running: "En curso",
  completed: "Completado",
  failed: "Falló",
  stopped: "Detenido",
  interrupted: "Interrumpido",
};
const KIND: Record<string, string> = { train: "Entrenamiento", eval: "Evaluación" };
const pct = (v: number) => (v * 100).toFixed(1) + " %";

/** Decodifica cualquier audio que entienda el navegador y lo devuelve como WAV mono 16 kHz en base64. */
async function toWav16kBase64(file: Blob): Promise<{ b64: string; seconds: number }> {
  const ctx = new AudioContext();
  const decoded = await ctx.decodeAudioData(await file.arrayBuffer());
  ctx.close();
  const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * 16000)), 16000);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const pcm = (await off.startRendering()).getChannelData(0);
  const wav = new DataView(new ArrayBuffer(44 + pcm.length * 2));
  const w = (o: number, s: string) => [...s].forEach((c, i) => wav.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF");
  wav.setUint32(4, 36 + pcm.length * 2, true);
  w(8, "WAVEfmt ");
  wav.setUint32(16, 16, true);
  wav.setUint16(20, 1, true);
  wav.setUint16(22, 1, true);
  wav.setUint32(24, 16000, true);
  wav.setUint32(28, 32000, true);
  wav.setUint16(32, 2, true);
  wav.setUint16(34, 16, true);
  w(36, "data");
  wav.setUint32(40, pcm.length * 2, true);
  pcm.forEach((v, i) => wav.setInt16(44 + i * 2, Math.max(-1, Math.min(1, v)) * 0x7fff, true));
  const bytes = new Uint8Array(wav.buffer);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return { b64: btoa(bin), seconds: decoded.duration };
}

export default function Audio({ pid }: { pid: number }) {
  const [tab, setTab] = useState<Tab>("Datos");
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [env, setEnv] = useState<{ ready: boolean; state: string } | null>(null);
  const [info, setInfo] = useState<{ models: any[]; languages: Record<string, string> }>({ models: [], languages: {} });

  const refresh = () => {
    api(`/projects/${pid}/datasets?kind=audio`).then(setDatasets).catch(() => {});
    api(`/projects/${pid}/jobs`)
      .then((j: Job[]) => setJobs(j.filter((x) => x.engine === "audio")))
      .catch(() => {});
    api("/engines/audio/env").then(setEnv).catch(() => {});
  };
  useEffect(() => {
    api("/audio/models").then(setInfo).catch(() => {});
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
          <h3>Motor de audio no instalado</h3>
          <p className="muted small">Instala PyTorch (CUDA), Transformers y las librerías de audio en un entorno aislado. Descarga varios GB.</p>
          <button
            className="primary"
            disabled={env.state === "installing"}
            onClick={() => api("/engines/audio/env/install", { method: "POST" }).then(refresh)}
          >
            {env.state === "installing" ? "Instalando… (puede tardar)" : "Instalar motor de audio"}
          </button>
          {env.state.startsWith("error") && <p className="msg">{env.state}</p>}
        </section>
      )}
      {tab === "Datos" && <Data pid={pid} datasets={datasets} refresh={refresh} />}
      {tab === "Entrenar" && <Train pid={pid} datasets={datasets} jobs={jobs} info={info} ready={!!env?.ready} refresh={refresh} />}
      {tab === "Evaluar" && <Evaluate pid={pid} datasets={datasets} trained={trained} info={info} />}
      {tab === "Transcribir" && <Transcribe trained={trained} info={info} />}
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
      const r = await api(`/projects/${pid}/datasets/audio`, { body: { name, source_dir: dir, license } });
      setMsg(`Importado: ${r.rows} clips (${r.report.minutes} min). ${r.report.invalid} descartados.`);
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
        <h3>Importar clips con transcripción</h3>
        <p className="muted small">
          La carpeta debe contener los audios (<code>.wav .flac .ogg .mp3</code>, de 0,3 a 30 s) y un <code>metadata.csv</code> con las columnas{" "}
          <code>file,text</code> (o un <code>metadata.jsonl</code>). Se valida cada clip y se guarda una copia en el proyecto.
        </p>
        <div className="row">
          <input value={dir} onChange={(e) => setDir(e.target.value)} placeholder="Carpeta con los clips y metadata.csv" />
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
        <h3>Datasets de audio</h3>
        {datasets.length === 0 && <p className="muted">Aún no hay datasets de audio.</p>}
        {datasets.map((d) => (
          <div key={d.id} className="item" style={{ flexDirection: "column", alignItems: "stretch" }}>
            <div>
              <b>{d.name}</b> · {d.rows} clips · {d.report?.minutes} min · prom. {d.report?.avg_seconds} s
              {d.license && <> · licencia: {d.license}</>}
            </div>
            {(d.report?.warnings ?? []).map((w: string) => (
              <div key={w} className="msg small">
                ⚠ {w}
              </div>
            ))}
            {Object.entries(d.report?.reasons ?? {}).map(([why, n]) => (
              <div key={why} className="small muted">
                Descartados ({String(n)}): {why}
              </div>
            ))}
          </div>
        ))}
      </section>
    </>
  );
}

function Train({ pid, datasets, jobs, info, ready, refresh }: any) {
  const [form, setForm] = useState<any>({
    dataset_id: 0, base_model: "openai/whisper-tiny", language: "spanish", epochs: 5, batch_size: 8, grad_accum: 1,
    learning_rate: 0.00001, val_split: 0.15, freeze_encoder: false, gradient_checkpointing: false,
  });
  const [license, setLicense] = useState("");
  const [sel, setSel] = useState<number | null>(null);
  const [metrics, setMetrics] = useState<any[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [msg, setMsg] = useState("");
  const cursor = useRef({ m: 0, l: 0 });

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
      setLicense((await api(`/models/license?repo_id=${encodeURIComponent(form.base_model)}`)).license);
    } catch (e: any) {
      setLicense(`no se pudo consultar (${e.message})`);
    }
  };
  const start = async (resume?: number) => {
    setMsg("");
    try {
      const r = await api(`/projects/${pid}/audio/train`, { body: { ...form, resume_job_id: resume ?? null } });
      setSel(r.job_id);
      refresh();
    } catch (e: any) {
      setMsg(e.message);
    }
  };

  const last = metrics[metrics.length - 1];
  const selJob = jobs.find((j: Job) => j.id === sel);

  return (
    <>
      <section className="card wide">
        <h3>Nuevo fine-tuning de Whisper</h3>
        <div className="grid">
          <label>
            Dataset
            <select value={form.dataset_id} onChange={(e) => set("dataset_id", Number(e.target.value))}>
              {datasets.length === 0 && <option value={0}>— importa un dataset primero —</option>}
              {datasets.map((d: Dataset) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.rows} clips)
                </option>
              ))}
            </select>
          </label>
          <label>
            Modelo base
            <div className="row">
              <select
                value={form.base_model}
                onChange={(e) => {
                  set("base_model", e.target.value);
                  setLicense("");
                }}
              >
                {info.models.map((m: any) => (
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
          <label>
            Idioma
            <select value={form.language} onChange={(e) => set("language", e.target.value)}>
              {Object.entries(info.languages).map(([k, v]) => (
                <option key={k} value={k}>
                  {String(v)}
                </option>
              ))}
            </select>
          </label>
          {num("epochs", "Épocas")}
          {num("batch_size", "Tamaño de lote")}
          {num("grad_accum", "Acumulación de gradiente")}
          {num("learning_rate", "Learning rate", 0.000005)}
          {num("val_split", "Fracción de validación", 0.05)}
          <label className="check">
            <input type="checkbox" checked={form.freeze_encoder} onChange={(e) => set("freeze_encoder", e.target.checked)} />
            Congelar el encoder (menos memoria, menos riesgo de sobreajuste)
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={form.gradient_checkpointing}
              onChange={(e) => set("gradient_checkpointing", e.target.checked)}
            />
            Gradient checkpointing (para Whisper small con poca VRAM)
          </label>
        </div>
        <button className="primary" disabled={!ready || !form.dataset_id} onClick={() => start()}>
          Iniciar entrenamiento
        </button>
        {msg && <p className="msg">{msg}</p>}
      </section>

      <section className="card wide">
        <h3>Trabajos de audio</h3>
        {jobs.length === 0 && <p className="muted">Aún no hay trabajos.</p>}
        {jobs.map((j: Job) => (
          <div key={j.id} className={"item" + (j.id === sel ? " sel" : "")}>
            <div onClick={() => setSel(j.id)} style={{ cursor: "pointer", flex: 1 }}>
              <b>
                #{j.id} {KIND[j.kind] ?? j.kind}
              </b>{" "}
              · <span className={"tag " + j.status}>{STATUS[j.status] ?? j.status}</span>
              {j.config.base_model && <> · {j.config.base_model}</>}
              {j.error && <div className="msg small">{j.error}</div>}
            </div>
            {["running", "queued"].includes(j.status) && (
              <button onClick={() => api(`/jobs/${j.id}/stop`, { method: "POST" }).then(refresh)}>Detener</button>
            )}
            {j.kind === "train" && ["stopped", "failed", "interrupted"].includes(j.status) && (
              <button onClick={() => start(j.id)}>Reanudar</button>
            )}
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
                WER <b>{pct(last.wer)}</b>
              </span>
              <span>
                CER <b>{pct(last.cer)}</b>
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

function Evaluate({ pid, datasets, trained, info }: any) {
  const [dsId, setDsId] = useState(0);
  const [picked, setPicked] = useState<Set<string>>(new Set(["base"]));
  const [baseModel, setBaseModel] = useState("openai/whisper-tiny");
  const [language, setLanguage] = useState("spanish");
  const [evals, setEvals] = useState<any[]>([]);
  const [msg, setMsg] = useState("");
  const [open_, setOpen] = useState<number | null>(null);

  const refresh = () => api(`/projects/${pid}/evals?engine=audio`).then(setEvals).catch(() => {});
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [pid]);
  useEffect(() => {
    if (!dsId && datasets[0]) setDsId(datasets[0].id);
  }, [datasets]);

  const toggle = (k: string) =>
    setPicked((s) => {
      const n = new Set(s);
      n.has(k) ? n.delete(k) : n.add(k);
      return n;
    });

  const run = async () => {
    setMsg("");
    try {
      const warn: string[] = [];
      for (const k of picked) {
        const r = await api(`/projects/${pid}/audio/eval`, {
          body: { dataset_id: dsId, job_id: k === "base" ? null : Number(k), base_model: baseModel, language },
        });
        if (r.contaminated) warn.push(`#${k}`);
      }
      if (warn.length) setMsg(`Aviso: ${warn.join(", ")} se entrenó con ese mismo dataset; el WER estará subestimado.`);
      refresh();
    } catch (e: any) {
      setMsg(e.message);
    }
  };

  const done = evals.filter((e) => e.result && e.dataset_id === dsId);
  const bestWer = done.length > 1 ? Math.min(...done.map((e) => e.result.wer)) : null;

  return (
    <>
      <section className="card wide">
        <h3>Nueva evaluación</h3>
        <p className="muted small">
          WER y CER sobre texto normalizado (sin mayúsculas ni puntuación); menos es mejor. Usa clips que <b>no</b> se hayan empleado en el
          entrenamiento.
        </p>
        <div className="grid">
          <label>
            Dataset de evaluación
            <select value={dsId} onChange={(e) => setDsId(Number(e.target.value))}>
              {datasets.length === 0 && <option value={0}>— sin datasets —</option>}
              {datasets.map((d: Dataset) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.rows} clips)
                </option>
              ))}
            </select>
          </label>
          <label>
            Idioma
            <select value={language} onChange={(e) => setLanguage(e.target.value)}>
              {Object.entries(info.languages).map(([k, v]) => (
                <option key={k} value={k}>
                  {String(v)}
                </option>
              ))}
            </select>
          </label>
          <label className="check">
            <input type="checkbox" checked={picked.has("base")} onChange={() => toggle("base")} />
            Modelo base
            <select value={baseModel} onChange={(e) => setBaseModel(e.target.value)}>
              {info.models.map((m: any) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
          {trained.map((j: Job) => (
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
                <th>WER ↓</th>
                <th>CER ↓</th>
                <th>RTF</th>
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
                    <td className={e.result && e.dataset_id === dsId && e.result.wer === bestWer ? "best" : ""}>
                      {e.result ? pct(e.result.wer) : "—"}
                    </td>
                    <td>{e.result ? pct(e.result.cer) : "—"}</td>
                    <td title="Tiempo de proceso / duración del audio (menos de 1 = más rápido que tiempo real)">
                      {e.result ? e.result.rtf : "—"}
                    </td>
                  </tr>
                  {open_ === e.id && e.result && (
                    <tr>
                      <td colSpan={6}>
                        <div className="small muted">
                          {e.result.rows} clips · {e.result.words} palabras · {e.result.audio_seconds} s de audio
                        </div>
                        {e.result.samples.map((s: any, i: number) => (
                          <div key={i} className="sample">
                            <div className="muted small">
                              {s.prompt} · WER {pct(s.wer)}
                            </div>
                            <div>
                              <b>Esperado:</b> {s.reference}
                            </div>
                            <div>
                              <b>Transcrito:</b> {s.prediction || "(vacío)"}
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
      </section>
    </>
  );
}

function Transcribe({ trained, info }: { trained: Job[]; info: any }) {
  const [choice, setChoice] = useState("base");
  const [baseModel, setBaseModel] = useState("openai/whisper-tiny");
  const [language, setLanguage] = useState("spanish");
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [result, setResult] = useState<{ text: string; audio_seconds: number; seconds: number } | null>(null);

  const load = async () => {
    setBusy(true);
    setErr("");
    try {
      const job = trained.find((j) => String(j.id) === choice);
      await api("/audio/load", { body: { job_id: job ? job.id : null, base_model: baseModel } });
      setLoaded(true);
      setResult(null);
    } catch (e: any) {
      setErr(e.message);
    }
    setBusy(false);
  };

  const onFile = async (f?: File) => {
    if (!f) return;
    setBusy(true);
    setErr("");
    setResult(null);
    try {
      setAudioUrl(URL.createObjectURL(f));
      const { b64 } = await toWav16kBase64(f);
      setResult(await api("/audio/transcribe", { body: { audio_b64: b64, language } }));
    } catch (e: any) {
      setErr(e.message || "No se pudo decodificar el audio.");
    }
    setBusy(false);
  };

  return (
    <section className="card wide">
      <h3>Transcribir audio</h3>
      <div className="row">
        <select value={choice} onChange={(e) => setChoice(e.target.value)}>
          <option value="base">Modelo base</option>
          {trained.map((j) => (
            <option key={j.id} value={j.id}>
              Entrenamiento #{j.id} · {j.config.base_model}
            </option>
          ))}
        </select>
        {choice === "base" && (
          <select value={baseModel} onChange={(e) => setBaseModel(e.target.value)}>
            {info.models.map((m: any) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        )}
        <select value={language} onChange={(e) => setLanguage(e.target.value)}>
          <option value="auto">Detectar idioma</option>
          {Object.entries(info.languages).map(([k, v]) => (
            <option key={k} value={k}>
              {String(v)}
            </option>
          ))}
        </select>
        <button className="primary" disabled={busy} onClick={load}>
          {loaded ? "Recargar" : "Cargar"}
        </button>
        {loaded && <button onClick={() => api("/audio/unload", { method: "POST" }).then(() => setLoaded(false))}>Descargar</button>}
      </div>
      <input type="file" accept="audio/*" disabled={!loaded || busy} onChange={(e) => onFile(e.target.files?.[0])} />
      {busy && <p className="muted small">Procesando…</p>}
      {err && <p className="msg">{err}</p>}
      {audioUrl && <audio controls src={audioUrl} style={{ display: "block", margin: "12px 0", width: "100%" }} />}
      {result && (
        <div>
          <div className="bubble assistant" style={{ maxWidth: "100%" }}>
            {result.text || "(sin texto)"}
          </div>
          <p className="muted small">
            {result.audio_seconds} s de audio transcritos en {result.seconds} s. Los audios de más de 28 s se cortan en ventanas, y puede
            partirse alguna palabra en el corte.
          </p>
        </div>
      )}
    </section>
  );
}
