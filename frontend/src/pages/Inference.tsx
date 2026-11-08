import { useEffect, useState } from "react";
import { api, Job } from "../api";

type Msg = { role: "user" | "assistant"; content: string };

export default function Inference({ pid }: { pid: number }) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [choice, setChoice] = useState("base");
  const [baseModel, setBaseModel] = useState("Qwen/Qwen2.5-0.5B-Instruct");
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [err, setErr] = useState("");
  const [temp, setTemp] = useState(0.7);

  useEffect(() => {
    api(`/projects/${pid}/jobs`)
      .then((j: Job[]) => setJobs(j.filter((x) => x.engine === "llm" && x.kind === "train" && x.status === "completed")))
      .catch(() => {});
  }, [pid]);

  const load = async () => {
    setBusy(true);
    setErr("");
    try {
      const job = jobs.find((j) => String(j.id) === choice);
      await api("/llm/load", {
        body: {
          base_model: job ? job.config.base_model : baseModel,
          adapter_path: job ? job.output_dir + "\\adapter" : null,
        },
      });
      setLoaded(true);
      setMsgs([]);
    } catch (e: any) {
      setErr(e.message);
    }
    setBusy(false);
  };

  const send = async () => {
    const next: Msg[] = [...msgs, { role: "user", content: input }];
    setMsgs(next);
    setInput("");
    setBusy(true);
    setErr("");
    try {
      const r = await api("/llm/chat", { body: { messages: next, temperature: temp } });
      setMsgs([...next, { role: "assistant", content: r.reply }]);
    } catch (e: any) {
      setErr(e.message);
    }
    setBusy(false);
  };

  return (
    <section className="card wide">
      <h3>Probar modelo</h3>
      <div className="row">
        <select value={choice} onChange={(e) => setChoice(e.target.value)}>
          <option value="base">Modelo base (sin adaptador)</option>
          {jobs.map((j) => (
            <option key={j.id} value={j.id}>
              Entrenamiento #{j.id} · {j.config.base_model}
            </option>
          ))}
        </select>
        {choice === "base" && <input value={baseModel} onChange={(e) => setBaseModel(e.target.value)} />}
        <button className="primary" disabled={busy} onClick={load}>
          {loaded ? "Recargar" : "Cargar"}
        </button>
        {loaded && (
          <button onClick={() => api("/llm/unload", { method: "POST" }).then(() => setLoaded(false))}>Descargar</button>
        )}
      </div>
      {loaded && (
        <label className="small muted">
          Temperatura {temp}
          <input type="range" min="0" max="1.5" step="0.1" value={temp} onChange={(e) => setTemp(Number(e.target.value))} />
        </label>
      )}
      <div className="chat">
        {msgs.map((m, i) => (
          <div key={i} className={"bubble " + m.role}>
            {m.content}
          </div>
        ))}
        {busy && <div className="muted small">Procesando…</div>}
      </div>
      {err && <p className="msg">{err}</p>}
      <div className="row">
        <input
          disabled={!loaded || busy}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && input.trim() && send()}
          placeholder={loaded ? "Escribe un mensaje…" : "Carga un modelo primero"}
        />
        <button disabled={!loaded || busy || !input.trim()} onClick={send}>
          Enviar
        </button>
      </div>
    </section>
  );
}
