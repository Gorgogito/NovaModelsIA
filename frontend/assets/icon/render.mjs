// Renderiza icon.svg a icon-1024.png (con esquinas transparentes) usando Edge, para que el PNG y el SVG sean idénticos.
// Uso (desde frontend/):  node assets/icon/render.mjs
// Después:                npx tauri icon assets/icon/icon-1024.png      (genera todos los tamaños y formatos)
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const HERE = dirname(fileURLToPath(import.meta.url));
const svg = readFileSync(join(HERE, "icon.svg"), "utf8");

const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await browser.newPage({ viewport: { width: 1024, height: 1024 }, deviceScaleFactor: 1 });
await page.setContent(`<!doctype html><body style="margin:0;background:transparent">${svg}</body>`);
await page.screenshot({ path: join(HERE, "icon-1024.png"), omitBackground: true, clip: { x: 0, y: 0, width: 1024, height: 1024 } });
await browser.close();
console.log("icon-1024.png generado");
