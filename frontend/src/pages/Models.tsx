import { useEffect, useState } from "react";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { api } from "../api";

type Model = {
  id: number;
  engine: string;
  status: string;
  base: string;
  created_at: number;
  output_dir: string;
  size_mb: number;
  artifacts: { name: string; path: string; size_mb: number }[];
  final: { step?: number; loss?: number; eval_loss?: number; val_acc?: number };
  hub_id: string;
};

const ENGINE: Record<string, string> = { llm: "LLM", vision: "Visión" };
const STATUS: Record<string, string> = {
  queued: "En cola",
  running: "En curso",
  completed: "Completado",
  failed: "Falló",
  stopped: "Detenido",
  interrupted: "Interrumpido",
};

export default function Models({ pid }: { pid: number }) {
  const [models, setModels] = useState<Model[]>([]);
  const [lic, setLic] = useState<Record<number, string>>({});
  const [confirm, setConfirm] = useState<number | null>(null);
  const [msg, setMsg] = useState("");

  const load = () => api(`/projects/${pid}/models`).then(setModels).catch((e) => setMsg(e.message));
  useEffect(() => {
    load();
  }, [pid]);

  const license = async (m: Model) => {
    try {
      const r = await api(`/models/license?repo_id=${encodeURIComponent(m.hub_id)}`);
      setLic((x) => ({ ...x, [m.id]: r.license }));
    } catch (e: any) {
      setLic((x) => ({ ...x, [m.id]: `no se pudo consultar (${e.message})` }));
    }
  };
  const reveal = async (path: string) => {
    try {
      await revealItemInDir(path);
    } catch {
      setMsg("Abrir carpetas solo funciona en la app de escritorio. Ruta: " + path);
    }
  };
  const remove = async (id: number) => {
    try {
      await api(`/jobs/${id}`, { method: "DELETE" });
      setConfirm(null);
      load();
    } catch (e: any) {
      setMsg(e.message);
    }
  };

  // Un entrenamiento reanudado comparte carpeta con el original: se cuenta una sola vez.
  const total = [...new Map(models.map((m) => [m.output_dir, m.size_mb])).values()].reduce((a, v) => a + v, 0);

  return (
    <section className="card wide">
      <h3>Modelos entrenados</h3>
      <p className="muted small">
        {models.length} modelo{models.length === 1 ? "" : "s"} · {(total / 1024).toFixed(2)} GB en disco. La licencia que se muestra es la del modelo
        base en el Hub; el modelo resultante hereda sus condiciones.
      </p>
      {msg && <p className="msg">{msg}</p>}
      {models.length === 0 && <p className="muted">Aún no hay modelos. Entrena uno en Entrenamiento o Visión.</p>}
      {models.map((m) => (
        <div key={m.id} className="item" style={{ flexDirection: "column", alignItems: "stretch" }}>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <div style={{ flex: 1 }}>
              <b>
                #{m.id} · {m.base}
              </b>{" "}
              <span className="tag">{ENGINE[m.engine] ?? m.engine}</span>{" "}
              <span className={"tag " + m.status}>{STATUS[m.status] ?? m.status}</span>
              <div className="small muted">
                {new Date(m.created_at * 1000).toLocaleString()} · {m.size_mb.toFixed(1)} MB
                {m.final.loss !== undefined && <> · loss {m.final.loss.toFixed(4)}</>}
                {m.final.val_acc !== undefined && <> · precisión val. {(m.final.val_acc * 100).toFixed(1)} %</>}
              </div>
            </div>
            <button onClick={() => license(m)}>Licencia</button>
            <button onClick={() => reveal(m.output_dir)}>Abrir carpeta</button>
            {confirm === m.id ? (
              <>
                <button className="primary" onClick={() => remove(m.id)}>
                  Confirmar borrado
                </button>
                <button onClick={() => setConfirm(null)}>Cancelar</button>
              </>
            ) : (
              <button disabled={["queued", "running"].includes(m.status)} onClick={() => setConfirm(m.id)}>
                Borrar
              </button>
            )}
          </div>
          {lic[m.id] && (
            <div className="small">
              Licencia de <code>{m.hub_id}</code>: <b>{lic[m.id]}</b>
            </div>
          )}
          {m.artifacts.map((a) => (
            <div key={a.name} className="small muted">
              📦 {a.name} · {a.size_mb} MB{" "}
              <a href="#" onClick={(e) => (e.preventDefault(), reveal(a.path))}>
                mostrar
              </a>
            </div>
          ))}
        </div>
      ))}
    </section>
  );
}
