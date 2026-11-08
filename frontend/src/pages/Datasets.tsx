import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { api, Dataset } from "../api";

export default function Datasets({ pid }: { pid: number }) {
  const [list, setList] = useState<Dataset[]>([]);
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [license, setLicense] = useState("");
  const [msg, setMsg] = useState("");
  const [preview, setPreview] = useState<any[] | null>(null);

  const load = () => api(`/projects/${pid}/datasets?kind=text`).then(setList).catch((e) => setMsg(e.message));
  useEffect(() => {
    load();
  }, [pid]);

  const pick = async () => {
    try {
      const f = await open({ filters: [{ name: "JSONL", extensions: ["jsonl", "json"] }] });
      if (typeof f === "string") {
        setPath(f);
        if (!name) setName(f.split(/[\\/]/).pop()!.replace(/\.\w+$/, ""));
      }
    } catch {
      setMsg("El selector de archivos solo funciona en la app de escritorio; escribe la ruta.");
    }
  };

  const doImport = async () => {
    setMsg("");
    try {
      const r = await api(`/projects/${pid}/datasets`, { body: { name, source_path: path, license } });
      setMsg(`Importado: ${r.rows} filas válidas (${r.report.invalid} inválidas, ${r.report.duplicates} duplicadas eliminadas).`);
      setPath("");
      setName("");
      load();
    } catch (e: any) {
      setMsg(e.message);
    }
  };

  return (
    <>
      <section className="card">
        <h3>Importar dataset (JSONL)</h3>
        <p className="muted small">
          Formatos: <code>messages</code> (chat), <code>instruction/output</code>, <code>prompt/completion</code>,{" "}
          <code>question/answer</code>. Se guarda una copia limpia, sin inválidos ni duplicados.
        </p>
        <div className="row">
          <input value={path} onChange={(e) => setPath(e.target.value)} placeholder="Ruta del archivo .jsonl" />
          <button onClick={pick}>Examinar…</button>
        </div>
        <div className="row">
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Nombre" />
          <input value={license} onChange={(e) => setLicense(e.target.value)} placeholder="Licencia del dataset (ej. MIT)" />
        </div>
        <button className="primary" disabled={!path || !name} onClick={doImport}>
          Importar y validar
        </button>
        {msg && <p className="msg">{msg}</p>}
      </section>
      <section className="card">
        <h3>Datasets del proyecto</h3>
        {list.length === 0 && <p className="muted">Aún no hay datasets.</p>}
        {list.map((d) => (
          <div key={d.id} className="item">
            <div style={{ flex: 1 }}>
              <b>{d.name}</b> · {d.rows} filas · prom. {d.report?.avg_chars} caracteres
              {d.license && <> · licencia: {d.license}</>}
            </div>
            <button onClick={async () => setPreview(await api(`/datasets/${d.id}/preview?limit=3`))}>Vista previa</button>
          </div>
        ))}
        {preview && <pre className="logs">{JSON.stringify(preview, null, 2)}</pre>}
      </section>
    </>
  );
}
