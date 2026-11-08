import { useEffect, useRef, useState } from "react";
import { api, Dataset, Job } from "../api";
import LossChart from "./LossChart";

const DEFAULTS = {
  base_model: "Qwen/Qwen2.5-0.5B-Instruct",
  epochs: 1,
  batch_size: 1,
  grad_accum: 8,
  learning_rate: 0.0002,
  max_seq_len: 512,
  lora_r: 16,
  lora_alpha: 32,
  optimizer: "paged_adamw_8bit",
  scheduler: "cosine",
  quant_4bit: true,
  gradient_checkpointing: true,
  save_steps: 50,
};
const STATUS: Record<string, string> = {
  queued: "En cola",
  running: "Entrenando",
  completed: "Completado",
  failed: "Falló",
  stopped: "Detenido",
  interrupted: "Interrumpido",
};

export default function Training({ pid }: { pid: number }) {
  const [env, setEnv] = useState<{ ready: boolean; state: string } | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [form, setForm] = useState<any>({ ...DEFAULTS, dataset_id: 0 });
  const [license, setLicense] = useState("");
  const [sel, setSel] = useState<number | null>(null);
  const [metrics, setMetrics] = useState<any[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [msg, setMsg] = useState("");
  const cursor = useRef({ m: 0, l: 0 });

  const refresh = () => {
    api("/engines/llm/env").then(setEnv).catch(() => {});
    api(`/projects/${pid}/jobs`).then((j: Job[]) => setJobs(j.filter((x) => x.engine === "llm"))).catch(() => {});
  };

  useEffect(() => {
    api(`/projects/${pid}/datasets?kind=text`)
      .then((d: Dataset[]) => {
        setDatasets(d);
        if (d[0]) setForm((f: any) => ({ ...f, dataset_id: f.dataset_id || d[0].id }));
      })
      .catch(() => {});
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [pid]);

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
        setMetrics((x) => [...x, ...m]);
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
      const r = await api(`/models/license?repo_id=${encodeURIComponent(form.base_model)}`);
      setLicense(r.license);
    } catch (e: any) {
      setLicense(`no se pudo consultar (${e.message})`);
    }
  };

  const start = async (resume?: number) => {
    setMsg("");
    try {
      const r = await api(`/projects/${pid}/train`, { body: { ...form, resume_job_id: resume ?? null } });
      setSel(r.job_id);
      refresh();
    } catch (e: any) {
      setMsg(e.message);
    }
  };

  const exportJob = async (id: number, gguf: boolean) => {
    try {
      const r = await api(`/projects/${pid}/export`, { body: { job_id: id, gguf } });
      setSel(r.job_id);
      refresh();
    } catch (e: any) {
      setMsg(e.message);
    }
  };

  // El último evento de Trainer es un resumen sin loss: se muestra la última métrica de entrenamiento real.
  const last = [...metrics].reverse().find((m) => m.loss !== undefined) ?? metrics[metrics.length - 1];
  const selJob = jobs.find((j) => j.id === sel);

  return (
    <>
      {env && !env.ready && (
        <section className="card warn">
          <h3>Motor LLM no instalado</h3>
          <p className="muted small">
            Instala PyTorch (CUDA), Transformers, PEFT, TRL y bitsandbytes en un entorno aislado. Descarga varios GB.
          </p>
          <button
            className="primary"
            disabled={env.state === "installing"}
            onClick={() => api("/engines/llm/env/install", { method: "POST" }).then(refresh)}
          >
            {env.state === "installing" ? "Instalando… (puede tardar)" : "Instalar motor LLM"}
          </button>
          {env.state.startsWith("error") && <p className="msg">{env.state}</p>}
        </section>
      )}

      <section className="card wide">
        <h3>Nuevo entrenamiento (LoRA / QLoRA)</h3>
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
            Modelo base (Hugging Face)
            <div className="row">
              <input
                value={form.base_model}
                onChange={(e) => {
                  set("base_model", e.target.value);
                  setLicense("");
                }}
              />
              <button onClick={checkLicense}>Licencia</button>
            </div>
            {license && (
              <span className="small muted">
                Licencia: <b>{license}</b>
              </span>
            )}
          </label>
          {num("epochs", "Épocas", 0.5)}
          {num("batch_size", "Tamaño de lote")}
          {num("grad_accum", "Acumulación de gradiente")}
          {num("learning_rate", "Learning rate", 0.00005)}
          {num("max_seq_len", "Longitud máx. de secuencia")}
          {num("lora_r", "LoRA rank (r)")}
          {num("lora_alpha", "LoRA alpha")}
          {num("save_steps", "Guardar cada N pasos")}
          <label>
            Optimizador
            <select value={form.optimizer} onChange={(e) => set("optimizer", e.target.value)}>
              {["paged_adamw_8bit", "adamw_torch", "adamw_8bit", "adafactor"].map((o) => (
                <option key={o}>{o}</option>
              ))}
            </select>
          </label>
          <label>
            Scheduler
            <select value={form.scheduler} onChange={(e) => set("scheduler", e.target.value)}>
              {["cosine", "linear", "constant"].map((o) => (
                <option key={o}>{o}</option>
              ))}
            </select>
          </label>
          <label className="check">
            <input type="checkbox" checked={form.quant_4bit} onChange={(e) => set("quant_4bit", e.target.checked)} />
            Cuantización 4-bit (QLoRA)
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={form.gradient_checkpointing}
              onChange={(e) => set("gradient_checkpointing", e.target.checked)}
            />
            Gradient checkpointing
          </label>
        </div>
        <button className="primary" disabled={!env?.ready || !form.dataset_id} onClick={() => start()}>
          Iniciar entrenamiento
        </button>
        {msg && <p className="msg">{msg}</p>}
      </section>

      <section className="card wide">
        <h3>Entrenamientos</h3>
        {jobs.length === 0 && <p className="muted">Aún no hay entrenamientos.</p>}
        {jobs.map((j) => (
          <div key={j.id} className={"item" + (j.id === sel ? " sel" : "")}>
            <div onClick={() => setSel(j.id)} style={{ cursor: "pointer", flex: 1 }}>
              <b>
                #{j.id} {j.kind === "train" ? "Entrenamiento" : "Exportación"}
              </b>{" "}
              · <span className={"tag " + j.status}>{STATUS[j.status] ?? j.status}</span>
              {j.kind === "train" && <> · {j.config.base_model}</>}
              {j.error && <div className="msg small">{j.error}</div>}
            </div>
            {["running", "queued"].includes(j.status) && (
              <button onClick={() => api(`/jobs/${j.id}/stop`, { method: "POST" }).then(refresh)}>Detener</button>
            )}
            {j.kind === "train" && ["stopped", "failed", "interrupted"].includes(j.status) && (
              <button onClick={() => start(j.id)}>Reanudar</button>
            )}
            {j.kind === "train" && j.status === "completed" && (
              <>
                <button onClick={() => exportJob(j.id, false)}>Exportar fusionado</button>
                <button onClick={() => exportJob(j.id, true)}>Exportar GGUF</button>
              </>
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
                Paso{" "}
                <b>
                  {last.step}/{last.max_steps}
                </b>
              </span>
              {last.loss !== undefined && (
                <span>
                  Loss <b>{last.loss.toFixed(4)}</b>
                </span>
              )}
              {last.eval_loss !== undefined && (
                <span>
                  Eval loss <b>{last.eval_loss.toFixed(4)}</b>
                </span>
              )}
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
