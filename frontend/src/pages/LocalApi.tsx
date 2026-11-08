import { useEffect, useState } from "react";
import { api } from "../api";

type Model = { id: string; nova: { modality: string; kind: string; base_model: string } };
type Info = {
  base_url: string;
  key_required: boolean;
  models: Model[];
  loaded: { modality: string; id: string } | null;
};

const MODALITY: Record<string, string> = { llm: "Texto (chat)", audio: "Audio (transcripción)", image: "Imagen" };
const KIND: Record<string, string> = { trained: "entrenado", base: "base" };

function Snippet({ title, code }: { title: string; code: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* el portapapeles puede no estar disponible: el texto sigue siendo seleccionable */
    }
  };
  return (
    <div style={{ marginBottom: 14 }}>
      <div className="row" style={{ marginBottom: 4 }}>
        <b className="small" style={{ flex: 1 }}>
          {title}
        </b>
        <button onClick={copy}>{copied ? "Copiado" : "Copiar"}</button>
      </div>
      <pre className="logs" style={{ marginTop: 0, maxHeight: "none" }}>
        {code}
      </pre>
    </div>
  );
}

export default function LocalApi() {
  const [info, setInfo] = useState<Info | null>(null);
  const [err, setErr] = useState("");

  const load = () => api("/api-info").then(setInfo).catch((e) => setErr(e.message));
  useEffect(() => {
    load();
    const id = setInterval(load, 3000);
    return () => clearInterval(id);
  }, []);

  if (!info) return <p className="muted">{err || "Cargando…"}</p>;

  const first = (m: string) => info.models.find((x) => x.nova.modality === m)?.id;
  const llm = first("llm") ?? "nova-llm-1";
  const audio = first("audio") ?? "openai/whisper-tiny";
  const image = first("image") ?? "nova-sd-lora-1";
  const auth = info.key_required ? `-H "Authorization: Bearer $NOVA_API_KEY" ` : "";
  const key = info.key_required ? 'os.environ["NOVA_API_KEY"]' : '"no-se-usa"';

  return (
    <>
      <section className="card wide">
        <h3>API local compatible con OpenAI</h3>
        <p className="muted small">
          Cualquier programa que hable con la API de OpenAI puede usar tus modelos sin internet ni coste. Solo escucha en tu equipo (127.0.0.1). Mientras
          la aplicación esté abierta, el servidor está activo.
        </p>
        <div className="row">
          <input readOnly value={info.base_url} onFocus={(e) => e.target.select()} />
          <button onClick={() => navigator.clipboard?.writeText(info.base_url).catch(() => {})}>Copiar URL</button>
        </div>
        <p className="small muted">
          Clave de API: {info.key_required ? "obligatoria (variable NOVA_API_KEY)" : "no necesaria. Para exigir una, arranca con la variable NOVA_API_KEY definida."}
        </p>
        <div className="item">
          <div style={{ flex: 1 }}>
            En la GPU ahora: <b>{info.loaded ? info.loaded.id : "nada"}</b>
            {info.loaded && <span className="muted small"> · {MODALITY[info.loaded.modality]}</span>}
          </div>
          <button disabled={!info.loaded} onClick={() => api("/api-info/unload", { method: "POST" }).then(load)}>
            Descargar de la GPU
          </button>
        </div>
        <p className="small muted">
          Solo hay un modelo cargado a la vez (tu GPU tiene 8 GB). Si una petición pide otro modelo, el actual se descarga y se carga el pedido: la
          primera respuesta tarda unos segundos de más. Mientras haya un entrenamiento o una evaluación en curso, la API responde que la GPU está ocupada.
        </p>
      </section>

      <section className="card wide">
        <h3>Modelos disponibles</h3>
        {info.models.length === 0 && <p className="muted">Aún no hay modelos. Entrena uno, o descarga un modelo base.</p>}
        <table className="tbl">
          <thead>
            <tr>
              <th>Id (campo "model")</th>
              <th>Tipo</th>
              <th>Origen</th>
              <th>Modelo base</th>
            </tr>
          </thead>
          <tbody>
            {info.models.map((m) => (
              <tr key={m.id}>
                <td>
                  <code>{m.id}</code>
                </td>
                <td>{MODALITY[m.nova.modality] ?? m.nova.modality}</td>
                <td>{KIND[m.nova.kind] ?? m.nova.kind}</td>
                <td className="small muted">{m.nova.kind === "trained" ? m.nova.base_model : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="small muted">
          Solo aparecen los modelos ya descargados o entrenados: la API nunca descarga nada por su cuenta. No se sirven los modelos de visión ni SDXL.
        </p>
      </section>

      <section className="card wide">
        <h3>Ejemplos</h3>
        <Snippet
          title="Chat (curl)"
          code={`curl ${info.base_url}/chat/completions ${auth}-H "Content-Type: application/json" -d '{
  "model": "${llm}",
  "messages": [{"role": "user", "content": "Hola"}],
  "max_tokens": 100
}'`}
        />
        <Snippet
          title="Chat con streaming (Python, SDK de OpenAI)"
          code={`from openai import OpenAI
client = OpenAI(base_url="${info.base_url}", api_key=${key})

for chunk in client.chat.completions.create(
        model="${llm}", stream=True,
        messages=[{"role": "user", "content": "Hola"}]):
    print(chunk.choices[0].delta.content or "", end="")`}
        />
        <Snippet
          title="Transcripción de audio (Python)"
          code={`with open("audio.wav", "rb") as f:   # wav, flac, ogg o mp3 (máx. 25 MB)
    r = client.audio.transcriptions.create(model="${audio}", file=f, language="es")
print(r.text)`}
        />
        <Snippet
          title="Generación de imágenes (Python)"
          code={`import base64
r = client.images.generate(
    model="${image}", prompt="a flat illustration of a red circle, novastyle",
    size="512x512", response_format="b64_json",
    extra_body={"seed": 1, "num_inference_steps": 25})   # extensiones propias, opcionales
open("imagen.png", "wb").write(base64.b64decode(r.data[0].b64_json))`}
        />
      </section>

      <section className="card wide">
        <h3>Qué soporta y qué no</h3>
        <ul className="small muted" style={{ lineHeight: 1.7 }}>
          <li>
            <b>Chat:</b> roles system, user y assistant; texto; <code>temperature</code>, <code>top_p</code>, <code>max_tokens</code>, <code>stream</code>. No hay{" "}
            <code>n&gt;1</code>, <code>stop</code>, herramientas ni imágenes de entrada.
          </li>
          <li>
            <b>Audio:</b> solo <code>json</code> y <code>text</code> como formato de respuesta; sin marcas de tiempo.
          </li>
          <li>
            <b>Imágenes:</b> solo <code>b64_json</code> (no se sirven URLs), de 128 a 768 px.
          </li>
          <li>El recuento de tokens de la respuesta es aproximado.</li>
        </ul>
      </section>
    </>
  );
}
