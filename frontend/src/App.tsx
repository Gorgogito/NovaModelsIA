import { useEffect, useState } from "react";
import "./App.css";
import { api, Project } from "./api";
import Datasets from "./pages/Datasets";
import Training from "./pages/Training";
import Inference from "./pages/Inference";
import Evaluation from "./pages/Evaluation";
import Vision from "./pages/Vision";
import Models from "./pages/Models";
import Audio from "./pages/Audio";
import ImageGen from "./pages/ImageGen";
import LocalApi from "./pages/LocalApi";

type Gpu = { name: string; vram_total_mb: number; vram_used_mb: number; utilization_pct: number };
type Hw = { cpu_pct: number; cpu_cores: number; ram_total_mb: number; ram_used_mb: number; gpus: Gpu[] };

const MODULES = ["Panel", "Datasets", "Entrenamiento", "Evaluación", "Inferencia", "Visión", "Audio", "Imagen", "Modelos", "API local"];
const LATER = new Set<string>();

function Meter({ label, used, total, unit }: { label: string; used: number; total: number; unit: string }) {
  const pct = total ? Math.round((used / total) * 100) : 0;
  return (
    <div className="meter">
      <div className="meter-head">
        <span>{label}</span>
        <span>
          {used.toLocaleString()} / {total.toLocaleString()} {unit} · {pct}%
        </span>
      </div>
      <div className="bar">
        <div className="fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function App() {
  const [active, setActive] = useState("Panel");
  const [hw, setHw] = useState<Hw | null>(null);
  const [online, setOnline] = useState(false);
  const [projects, setProjects] = useState<Project[]>([]);
  const [pid, setPid] = useState<number | null>(null);
  const [newName, setNewName] = useState("");
  const [err, setErr] = useState("");

  const loadProjects = async () => {
    const p: Project[] = await api("/projects");
    setProjects(p);
    setPid((cur) => cur ?? p[0]?.id ?? null);
  };

  useEffect(() => {
    const tick = async () => {
      try {
        setHw(await api("/hardware"));
        setOnline(true);
        if (projects.length === 0) loadProjects().catch(() => {});
      } catch {
        setOnline(false);
      }
    };
    tick();
    const id = setInterval(tick, 2000);
    return () => clearInterval(id);
  }, [projects.length]);

  const createProject = async () => {
    setErr("");
    try {
      const p: Project = await api("/projects", { body: { name: newName } });
      setNewName("");
      setPid(p.id);
      await loadProjects();
    } catch (e: any) {
      setErr(e.message);
    }
  };

  const needsProject = ["Datasets", "Entrenamiento", "Evaluación", "Inferencia", "Visión", "Audio", "Imagen", "Modelos", "API local"].includes(active);

  return (
    <div className="shell">
      <aside className="sidebar">
        <h1>
          Nova<span>Models</span>
        </h1>
        <select className="proj" value={pid ?? ""} onChange={(e) => setPid(Number(e.target.value))}>
          {projects.length === 0 && <option value="">Sin proyectos</option>}
          {projects.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
        {MODULES.map((m) => (
          <button key={m} className={m === active ? "nav active" : "nav"} onClick={() => setActive(m)}>
            {m}
          </button>
        ))}
        <div className={online ? "status on" : "status off"}>
          {online ? "Backend conectado" : "Backend desconectado"}
        </div>
      </aside>
      <main className="content">
        <h2>{active}</h2>
        {LATER.has(active) && <p className="muted">Este módulo llega en una fase posterior.</p>}

        {needsProject && pid == null && <p className="muted">Crea un proyecto en el Panel para empezar.</p>}
        {active === "Datasets" && pid != null && <Datasets pid={pid} />}
        {active === "Entrenamiento" && pid != null && <Training pid={pid} />}
        {active === "Evaluación" && pid != null && <Evaluation pid={pid} />}
        {active === "Inferencia" && pid != null && <Inference pid={pid} />}
        {active === "Visión" && pid != null && <Vision pid={pid} />}
        {active === "Audio" && pid != null && <Audio pid={pid} />}
        {active === "Imagen" && pid != null && <ImageGen pid={pid} />}
        {active === "Modelos" && pid != null && <Models pid={pid} />}
        {active === "API local" && <LocalApi />}

        {active === "Panel" && (
          <>
            <section className="card">
              <h3>Proyectos</h3>
              <div className="row">
                <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="Nombre del nuevo proyecto" />
                <button className="primary" disabled={!newName.trim() || !online} onClick={createProject}>
                  Crear
                </button>
              </div>
              {err && <p className="msg">{err}</p>}
              {projects.map((p) => (
                <div key={p.id} className={"item" + (p.id === pid ? " sel" : "")}>
                  <div style={{ flex: 1 }}>{p.name}</div>
                  <button onClick={() => setPid(p.id)}>Abrir</button>
                </div>
              ))}
            </section>
            <section className="card">
              <h3>Recursos del sistema</h3>
              {!hw && <p className="muted">Esperando al backend…</p>}
              {hw && (
                <>
                  <Meter label={`CPU (${hw.cpu_cores} hilos)`} used={Math.round(hw.cpu_pct)} total={100} unit="%" />
                  <Meter label="RAM" used={hw.ram_used_mb} total={hw.ram_total_mb} unit="MB" />
                  {hw.gpus.map((g) => (
                    <div key={g.name}>
                      <Meter label={`VRAM · ${g.name}`} used={g.vram_used_mb} total={g.vram_total_mb} unit="MB" />
                      <Meter label="Uso de GPU" used={g.utilization_pct} total={100} unit="%" />
                    </div>
                  ))}
                </>
              )}
            </section>
          </>
        )}
      </main>
    </div>
  );
}
