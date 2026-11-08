// Construye el instalador de NovaModelsIA con el backend empaquetado incluido.
//
// Uso (desde frontend/):  npm run build:app
// Requiere haber construido antes el backend:  cd backend && python -m uv run python build_exe.py
//
// El recurso del backend vive en tauri.bundle.conf.json, no en tauri.conf.json: así `tauri dev` y
// `cargo check` funcionan en un clon limpio, sin construir antes el backend.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const exe = join(root, "..", "backend", "dist", "nova-backend", "nova-backend.exe");

if (!existsSync(exe)) {
  console.error(`\nNo se encontró el backend empaquetado:\n  ${exe}\n\nConstrúyelo primero:\n  cd backend\n  python -m uv run python build_exe.py\n`);
  process.exit(1);
}

const r = spawnSync("npx", ["tauri", "build", "--config", "src-tauri/tauri.bundle.conf.json"], { cwd: root, stdio: "inherit", shell: true });
process.exit(r.status ?? 1);
