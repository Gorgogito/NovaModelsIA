// Recorre cada pantalla de NovaModelsIA con Edge (Playwright), hace capturas y registra errores.
//
// Uso:  npm run tour            (desde frontend/)  -> la interfaz en Edge
//       npm run tour:tauri      (desde frontend/)  -> la ventana REAL de Tauri (webview), vía depuración remota
//       TAURI_EXE=ruta\a\app.exe npm run tour:tauri   -> prueba un ejecutable de release en vez de `tauri dev`
// Salida: e2e/screenshots/*.png, e2e/screenshots/report.json y report.html
//
// Arranca el backend y el servidor de Vite si no están corriendo, y solo detiene lo que arrancó.
// Es una prueba de la interfaz React en un navegador: no abre la ventana nativa de Tauri,
// así que los diálogos del sistema (selector de archivos, abrir carpeta) no se ejercitan.
import { spawn, execSync } from "node:child_process";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const HERE = dirname(fileURLToPath(import.meta.url));
const TAURI = process.argv.includes("--tauri");
const TAURI_EXE = process.env.TAURI_EXE;
const CDP = "http://127.0.0.1:9222";
const OUT = join(HERE, TAURI ? "screenshots-tauri" : "screenshots");
const APP = "http://localhost:1420";
const API = "http://127.0.0.1:8765";
const started = [];

async function up(url) {
  try {
    return (await fetch(url, { signal: AbortSignal.timeout(1500) })).ok;
  } catch {
    return false;
  }
}

async function ensure(name, url, cmd, args, cwd) {
  if (await up(url)) {
    console.log(`· ${name}: ya estaba corriendo`);
    return;
  }
  console.log(`· ${name}: arrancando…`);
  const p = spawn(cmd, args, { cwd, shell: true, stdio: "ignore", windowsHide: true });
  started.push(p);
  for (let i = 0; i < 60; i++) {
    if (await up(url)) return;
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error(`${name} no respondió en 60 s`);
}

function killStarted() {
  for (const p of started) {
    try {
      execSync(`taskkill /PID ${p.pid} /T /F`, { stdio: "ignore" });
    } catch {}
  }
}

rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

const results = [];
const issues = []; // errores de consola / peticiones fallidas, con la pantalla donde ocurrieron
let current = "inicio";
let tauriInfo = null;
let shotN = 0;

async function main() {
  let browser, page;
  if (TAURI) {
    // WebView2 admite depuración remota: Playwright se conecta al webview real de la ventana de Tauri.
    if (await up(`${CDP}/json/version`)) throw new Error("Ya hay algo escuchando en el puerto 9222 (¿otra instancia de la app abierta?). Ciérrala.");
    if (await up(`${API}/health`)) throw new Error("Ya hay un backend en el puerto 8765: ciérralo para comprobar que la app lo arranca sola.");
    const env = { ...process.env, WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: "--remote-debugging-port=9222" };
    const child = TAURI_EXE
      ? spawn(TAURI_EXE, [], { env, stdio: "ignore" })
      : spawn("npm", ["run", "tauri", "dev"], { cwd: join(HERE, ".."), env, shell: true, stdio: "ignore" });
    started.push(child);
    console.log(TAURI_EXE ? `· app: ${TAURI_EXE}` : "· app: npm run tauri dev (la primera compilación puede tardar minutos)");
    for (let i = 0; i < 900 && !(await up(`${CDP}/json/version`)); i++) await new Promise((r) => setTimeout(r, 1000));
    browser = await chromium.connectOverCDP(CDP);
    for (let i = 0; i < 90 && !page; i++) {
      page = browser.contexts().flatMap((c) => c.pages()).find((x) => /^https?:\/\/(localhost:1420|tauri\.localhost)/.test(x.url()));
      if (!page) await new Promise((r) => setTimeout(r, 1000));
    }
    if (!page) throw new Error("No se encontró la página de la app en el webview.");
    for (let i = 0; i < 90 && !(await up(`${API}/health`)); i++) await new Promise((r) => setTimeout(r, 1000));
    if (!(await up(`${API}/health`))) throw new Error("La app no arrancó el backend.");
    console.log("· backend arrancado por la propia app");
    try {  // ¿es el ejecutable empaquetado o el Python del entorno de desarrollo?
      const n = execSync(`powershell -NoProfile -Command "@(Get-Process nova-backend -ErrorAction SilentlyContinue).Count"`, { encoding: "utf8" }).trim();
      const dev = execSync(`powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '-m nova' }).Count"`, { encoding: "utf8" }).trim();
      console.log(`· procesos nova-backend.exe (empaquetado): ${n} · procesos "python -m nova" (desarrollo): ${dev}`);
    } catch {}
  } else {
    const backendDir = join(HERE, "..", "..", "backend");
    await ensure("backend", `${API}/health`, join(backendDir, ".venv", "Scripts", "python.exe"), ["-m", "nova"], backendDir);
    await ensure("interfaz (vite)", APP, "npm", ["run", "dev"], join(HERE, ".."));
    browser = await chromium.launch({ channel: "msedge", headless: true });
    page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
  }

  const projects = await (await fetch(`${API}/projects`)).json();
  console.log(`· proyectos: ${projects.map((p) => p.name).join(", ") || "(ninguno)"}`);


  page.on("console", (m) => m.type() === "error" && issues.push({ screen: current, kind: "consola", text: m.text().slice(0, 300) }));
  page.on("pageerror", (e) => issues.push({ screen: current, kind: "excepción", text: String(e).slice(0, 300) }));
  page.on("requestfailed", (r) => issues.push({ screen: current, kind: "petición fallida", text: `${r.method()} ${r.url()} — ${r.failure()?.errorText}` }));
  page.on("response", (r) => r.status() >= 400 && issues.push({ screen: current, kind: `HTTP ${r.status()}`, text: `${r.request().method()} ${r.url()}` }));

  async function shot(title, note = "") {
    const file = `${String(++shotN).padStart(2, "0")}-${title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.png`;
    await page.waitForTimeout(700);
    // El contenido se desplaza dentro de .content (altura fija): se agranda la ventana para capturarlo entero.
    const h = await page.evaluate(() => Math.max(document.querySelector(".content")?.scrollHeight ?? 0, 900));
    await page.setViewportSize({ width: 1280, height: Math.min(h, 7000) });
    await page.waitForTimeout(250);
    await page.screenshot({ path: join(OUT, file) });
    const overflowX = await page.evaluate(() => {
      const c = document.querySelector(".content");
      return document.documentElement.scrollWidth > window.innerWidth + 2 || (c ? c.scrollWidth > c.clientWidth + 2 : false);
    });
    await page.setViewportSize({ width: 1280, height: 900 });
    results.push({ title, file, note, overflowX });
    console.log(`  📸 ${file}`);
  }
  async function nav(name) {
    current = name;
    await page.locator(".nav", { hasText: new RegExp(`^${name}$`) }).click();
    await page.waitForTimeout(500);
  }
  async function pickProject(...names) {
    // Las listas llegan ordenadas por id descendente: el primer nombre coincidente es el más reciente.
    for (const n of names) {
      const opt = projects.find((p) => p.name === n);
      if (opt) {
        await page.locator("select.proj").selectOption(String(opt.id));
        await page.waitForTimeout(400);
        return n;
      }
    }
    return projects[0]?.name ?? "(sin proyecto)";
  }
  async function tab(name) {
    await page.locator(".tab", { hasText: new RegExp(`^${name}$`) }).click();
    await page.waitForTimeout(600);
  }
  const clickIf = async (loc) => {
    if ((await loc.count()) > 0) {
      await loc.first().click();
      await page.waitForTimeout(900);
      return true;
    }
    return false;
  };

  if (!TAURI) await page.goto(APP);
  await page.waitForSelector(".sidebar");
  await page.waitForTimeout(1500);

  // Panel
  await nav("Panel");
  await shot("Panel", "proyectos y recursos del sistema");

  // LLM
  const llm = await pickProject("E2E demo");
  await nav("Datasets");
  await shot("Datasets LLM", `proyecto: ${llm}`);
  if (await clickIf(page.getByRole("button", { name: "Vista previa" }))) await shot("Datasets LLM vista previa");

  await nav("Entrenamiento");
  await shot("Entrenamiento formulario");
  if (await clickIf(page.locator(".item div[style*='pointer']"))) await shot("Entrenamiento monitoreo", "curva de pérdida y registros");

  await nav("Evaluación");
  await shot("Evaluacion LLM");
  if (await clickIf(page.locator(".tbl tbody tr").first())) await shot("Evaluacion LLM ejemplos");

  await nav("Inferencia");
  await shot("Inferencia", "sin cargar modelo (no se usa la GPU)");

  // Visión
  const vis = await pickProject("Visión demo");
  await nav("Visión");
  await shot("Vision datos", `proyecto: ${vis}`);
  await tab("Entrenar");
  await shot("Vision entrenar");
  if (await clickIf(page.locator(".item div[style*='pointer']"))) await shot("Vision monitoreo");
  await tab("Evaluar");
  await shot("Vision evaluar");
  if (await clickIf(page.locator(".tbl tbody tr").first())) await shot("Vision matriz de confusion");
  await tab("Probar");
  await shot("Vision probar");

  // Audio
  const aud = await pickProject("Audio demo 2", "Audio demo");
  await nav("Audio");
  await shot("Audio datos", `proyecto: ${aud}`);
  await tab("Entrenar");
  await shot("Audio entrenar");
  if (await clickIf(page.locator(".item div[style*='pointer']"))) await shot("Audio monitoreo");
  await tab("Evaluar");
  await shot("Audio evaluar");
  if (await clickIf(page.locator(".tbl tbody tr").first())) await shot("Audio ejemplos");
  await tab("Transcribir");
  await shot("Audio transcribir");

  // Imagen
  const img = await pickProject("Imagen demo");
  await nav("Imagen");
  await shot("Imagen datos", `proyecto: ${img}`);
  await tab("Entrenar");
  await shot("Imagen entrenar");
  // El primer trabajo de la lista es la comparación; el monitoreo es del entrenamiento.
  if (await clickIf(page.locator(".item div[style*='pointer']", { hasText: "Entrenamiento LoRA" }))) await shot("Imagen monitoreo", "gráfica y muestras durante el entrenamiento");
  await tab("Generar");
  await shot("Imagen generar", "sin cargar modelo (no se usa la GPU)");
  await tab("Comparar");
  await shot("Imagen comparar", "modelo base frente a LoRA con la misma semilla");

  // API local (no depende del proyecto)
  await nav("API local");
  await shot("API local", "URL, modelos servidos y ejemplos");

  // Modelos (un proyecto con LLM y otro con visión)
  await pickProject("E2E demo");
  await nav("Modelos");
  await shot("Modelos LLM");
  await pickProject("Visión demo");
  await shot("Modelos vision");

  if (TAURI) {
    current = "tauri";
    tauriInfo = await page.evaluate(() => ({ origen: location.origin, internals: typeof window.__TAURI_INTERNALS__ }));
    console.log(`  · origen de la página: ${tauriInfo.origen} · APIs de Tauri: ${tauriInfo.internals}`);
    // "Abrir carpeta": comprueba que el permiso del plugin opener funciona. Abre el Explorador; se cierra solo
    // la ventana que se abre aquí (no las que ya tuvieras).
    const hwnds = () => {
      try {
        return execSync(`powershell -NoProfile -Command "(New-Object -ComObject Shell.Application).Windows() | ForEach-Object { $_.HWND }"`, { encoding: "utf8" }).split(/\s+/).filter(Boolean);
      } catch { return []; }
    };
    const before = new Set(hwnds());
    tauriInfo.revealItemInDir = await page.evaluate(async (path) => {
      try { await window.__TAURI_INTERNALS__.invoke("plugin:opener|reveal_item_in_dir", { paths: [path] }); return "ok"; }
      catch (e) { return String(e); }
    }, resolve(HERE, "..", "..", "data", "projects"));
    console.log(`  · permiso "Abrir carpeta" (reveal_item_in_dir): ${tauriInfo.revealItemInDir}`);
    await new Promise((r) => setTimeout(r, 2500));
    for (const h of hwnds().filter((x) => !before.has(x))) {
      try { execSync(`powershell -NoProfile -Command "(New-Object -ComObject Shell.Application).Windows() | Where-Object { $_.HWND -eq ${h} } | ForEach-Object { $_.Quit() }"`, { stdio: "ignore" }); } catch {}
    }
    if (tauriInfo.revealItemInDir !== "ok") issues.push({ screen: "tauri", kind: "permiso", text: `reveal_item_in_dir: ${tauriInfo.revealItemInDir}` });
  }

  await browser.close();
}

let failed = false;
try {
  await main();
} catch (e) {
  failed = true;
  console.error("✗ El recorrido se interrumpió en:", current, "\n ", e.message.split("\n")[0]);
} finally {
  killStarted();
}

const unique = [...new Map(issues.map((i) => [`${i.screen}|${i.kind}|${i.text}`, i])).values()];
writeFileSync(join(OUT, "report.json"), JSON.stringify({ tauri: tauriInfo, results, issues: unique }, null, 2));
writeFileSync(
  join(OUT, "report.html"),
  `<!doctype html><meta charset="utf-8"><title>Recorrido NovaModelsIA</title>
<style>body{font-family:system-ui;background:#0f1115;color:#e6e8ee;margin:24px}img{max-width:100%;border:1px solid #333;border-radius:8px}
h2{margin-top:32px}.bad{color:#e8a43a}.note{color:#8a91a3}</style>
<h1>Recorrido NovaModelsIA</h1>
<h2>Problemas detectados: ${unique.length}</h2>
${unique.map((i) => `<div class="bad">[${i.screen}] ${i.kind}: ${i.text.replace(/</g, "&lt;")}</div>`).join("") || "<div>Ninguno</div>"}
${results.map((r) => `<h2>${r.title}${r.overflowX ? ' <span class="bad">· desborda horizontalmente</span>' : ""}</h2><div class="note">${r.note}</div><img src="${r.file}">`).join("")}`
);

console.log(`\n${results.length} capturas · ${unique.length} problemas · ${results.filter((r) => r.overflowX).length} pantallas con desborde horizontal`);
unique.forEach((i) => console.log(`  ⚠ [${i.screen}] ${i.kind}: ${i.text}`));
console.log(`Informe: ${join(OUT, "report.html")}`);
process.exit(failed || unique.length ? 1 : 0);
