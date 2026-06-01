const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const pptxgen = require("pptxgenjs");
const { chromium } = require("playwright");
const { PDFDocument } = require("pdf-lib");

const ROOT = path.resolve(__dirname, "..");
const CAPTURE_DIR = path.join(__dirname, "generated", "app-slide-captures");
const PPTX_OUT = path.join(__dirname, "Behavioral_Fraud_Analytics_App_Rendered.pptx");
const PDF_OUT = path.join(__dirname, "Behavioral_Fraud_Analytics_App_Rendered.pdf");
const PORT = 4174;
const BASE_URL = `http://127.0.0.1:${PORT}`;
const SLIDE_W = 1152;
const SLIDE_H = 648;
const PPT_W = 13.333333;
const PPT_H = 7.5;

// React app slide indexes. S20_LiveDemo is intentionally omitted.
const SLIDES = [
  0, 1, 2, 3, 4,
  5, 6, 7, 8, 9,
  10, 11, 12, 13, 14,
  15, 16, 17, 18, 20,
];

function waitForServer(url, timeoutMs = 30_000) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const check = () => {
      fetch(url)
        .then((res) => {
          if (res.ok) resolve();
          else throw new Error(`HTTP ${res.status}`);
        })
        .catch((err) => {
          if (Date.now() - started > timeoutMs) {
            reject(new Error(`Timed out waiting for Vite at ${url}: ${err.message}`));
            return;
          }
          setTimeout(check, 250);
        });
    };
    check();
  });
}

async function captureSlides() {
  fs.rmSync(CAPTURE_DIR, { recursive: true, force: true });
  fs.mkdirSync(CAPTURE_DIR, { recursive: true });

  const server = spawn(
    "npm",
    ["--prefix", "presentation", "run", "dev", "--", "--host", "127.0.0.1", "--port", String(PORT)],
    { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] },
  );

  let serverOutput = "";
  server.stdout.on("data", (chunk) => {
    serverOutput += chunk.toString();
  });
  server.stderr.on("data", (chunk) => {
    serverOutput += chunk.toString();
  });

  try {
    await waitForServer(BASE_URL);

    const browser = await chromium.launch();
    const page = await browser.newPage({
      viewport: { width: SLIDE_W, height: SLIDE_H },
      deviceScaleFactor: 2,
    });

    for (let i = 0; i < SLIDES.length; i += 1) {
      const appSlide = SLIDES[i];
      const file = path.join(CAPTURE_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`);
      await page.goto(`${BASE_URL}/?pptxExport=1&slide=${appSlide}`, { waitUntil: "networkidle" });
      await page.waitForTimeout(900);
      await page.screenshot({ path: file, fullPage: false });
      console.log(`Captured slide ${i + 1}/${SLIDES.length}`);
    }

    await browser.close();
  } catch (err) {
    console.error(serverOutput);
    throw err;
  } finally {
    server.kill("SIGTERM");
    setTimeout(() => {
      if (!server.killed) server.kill("SIGKILL");
    }, 1000);
  }
}

async function buildPptx() {
  const pptx = new pptxgen();
  pptx.defineLayout({ name: "APP_16_9", width: PPT_W, height: PPT_H });
  pptx.layout = "APP_16_9";
  pptx.author = "Koutsompinas Konstantinos";
  pptx.company = "National and Kapodistrian University of Athens";
  pptx.subject = "Behavioral Fraud Analytics";
  pptx.title = "Behavioral Fraud Analytics";
  pptx.lang = "en-US";
  pptx.theme = {
    headFontFace: "Aptos Display",
    bodyFontFace: "Aptos",
    lang: "en-US",
  };

  for (let i = 0; i < SLIDES.length; i += 1) {
    const slide = pptx.addSlide();
    const image = path.join(CAPTURE_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`);
    slide.background = { color: "0A1628" };
    slide.addImage({ path: image, x: 0, y: 0, w: PPT_W, h: PPT_H });
  }

  await pptx.writeFile({ fileName: PPTX_OUT });
  console.log(`Saved ${PPTX_OUT}`);
}

async function buildPdf() {
  const pdf = await PDFDocument.create();
  pdf.setTitle("Behavioral Fraud Analytics");
  pdf.setAuthor("Koutsompinas Konstantinos");
  pdf.setSubject("Behavioral Fraud Analytics");
  pdf.setCreator("docs/build-ppt.js");
  pdf.setProducer("pdf-lib");

  for (let i = 0; i < SLIDES.length; i += 1) {
    const image = path.join(CAPTURE_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`);
    const png = await pdf.embedPng(fs.readFileSync(image));
    const page = pdf.addPage([SLIDE_W, SLIDE_H]);
    page.drawImage(png, { x: 0, y: 0, width: SLIDE_W, height: SLIDE_H });
  }

  fs.writeFileSync(PDF_OUT, await pdf.save());
  console.log(`Saved ${PDF_OUT}`);
}

async function main() {
  await captureSlides();
  await buildPptx();
  await buildPdf();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
