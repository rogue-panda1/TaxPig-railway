const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { execSync, execFileSync } = require('child_process');
const { createCanvas, loadImage } = require('@napi-rs/canvas');
const { ImageAnnotatorClient } = require('@google-cloud/vision');

const PORT = Number(process.env.PORT) || 3000;
const MAX_MB = Number(process.env.MAX_MB) || 25;
const MAX_BYTES = MAX_MB * 1024 * 1024;
const PDF_RENDER_TIMEOUT_MS = Number(process.env.PDF_RENDER_TIMEOUT_MS) || 120000;
const OFFICE_TIMEOUT_MS = Number(process.env.OFFICE_TIMEOUT_MS) || 120000;
const PDF_RENDER_DPI = Number(process.env.PDF_RENDER_DPI) || 200;
const PDF_MAX_PIXELS = Number(process.env.PDF_MAX_PIXELS) || 40000000;
const PDF_RENDERERS = (process.env.PDF_RENDERERS || 'pdftocairo,ghostscript,pdfjs')
  .split(',')
  .map((r) => r.trim().toLowerCase())
  .filter(Boolean);
const ALLOWED_TYPES = process.env.ALLOWED_TYPES
  ? process.env.ALLOWED_TYPES.split(',').map((t) => t.trim().toLowerCase())
  : null;

const app = express();

// Keep the service alive even if a dependency emits an unhandled rejection.
// (Google auth can do this when ADC is not configured.)
process.on('unhandledRejection', (reason) => {
  console.error('UnhandledRejection:', reason);
});
process.on('uncaughtException', (err) => {
  console.error('UncaughtException:', err);
});

// Optional: allow passing service account JSON via env (Railway-friendly).
// - If it starts with "{", treat as raw JSON
// - Else treat as base64-encoded JSON
function maybeConfigureGoogleCredentialsFromEnv() {
  try {
    if (process.env.GOOGLE_APPLICATION_CREDENTIALS) return null;
    const raw = String(process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON || '').trim();
    if (!raw) return null;
    let jsonText = raw;
    if (!raw.startsWith('{')) {
      jsonText = Buffer.from(raw, 'base64').toString('utf8');
    }
    // Validate JSON before writing.
    const parsed = JSON.parse(jsonText);
    const p = path.join(os.tmpdir(), 'google_application_credentials.json');
    fs.writeFileSync(p, JSON.stringify(parsed), 'utf8');
    process.env.GOOGLE_APPLICATION_CREDENTIALS = p;
    return null;
  } catch (e) {
    return `google_adc_env_error:${e?.message || String(e)}`;
  }
}

const GOOGLE_ADC_ENV_ERROR = maybeConfigureGoogleCredentialsFromEnv();

const memoryStorage = multer.memoryStorage();
const upload = multer({
  storage: memoryStorage,
  limits: { fileSize: MAX_BYTES },
});

const rawParser = express.raw({
  type: (req) => !(req.get('Content-Type') || '').includes('multipart/form-data'),
  limit: `${MAX_MB}mb`,
});

function getInputBuffer(req) {
  if (req.file && req.file.buffer) return req.file.buffer;
  if (Buffer.isBuffer(req.body) && req.body.length) return req.body;
  return null;
}

function checkAllowedType(filename, forOffice = false) {
  if (!ALLOWED_TYPES) return true;
  const ext = path.extname(filename || '').slice(1).toLowerCase();
  const types = ALLOWED_TYPES.map((t) => t.replace(/^\./, ''));
  return types.includes(ext);
}

function setMetadataHeaders(res, meta) {
  res.setHeader('X-Conversion-Metadata', Buffer.from(JSON.stringify(meta)).toString('base64'));
  res.setHeader('X-Page-Count', String(meta.pageCount ?? 1));
  res.setHeader('X-Page-Number', String(meta.pageNumber ?? 1));
  res.setHeader('X-Width', String(meta.width ?? 0));
  res.setHeader('X-Height', String(meta.height ?? 0));
  if (meta.renderer) res.setHeader('X-Renderer', String(meta.renderer));
}

function wantsJsonResponse(req) {
  const queryResponse = String(req.query.response || req.query.format || '').toLowerCase();
  const headerResponse = String(req.get('X-Response-Format') || '').toLowerCase();
  const accept = String(req.get('Accept') || '').toLowerCase();
  return queryResponse === 'json' || headerResponse === 'json' || accept.includes('application/json');
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function bboxUnion(a, b) {
  if (!a) return b;
  if (!b) return a;
  return {
    x0: Math.min(a.x0, b.x0),
    y0: Math.min(a.y0, b.y0),
    x1: Math.max(a.x1, b.x1),
    y1: Math.max(a.y1, b.y1),
  };
}

function bboxFromVertices(vertices = []) {
  const xs = vertices.map((v) => Number(v.x ?? 0));
  const ys = vertices.map((v) => Number(v.y ?? 0));
  return {
    x0: Math.min(...xs),
    y0: Math.min(...ys),
    x1: Math.max(...xs),
    y1: Math.max(...ys),
  };
}

function median(nums) {
  const a = nums.filter((n) => Number.isFinite(n)).slice().sort((x, y) => x - y);
  if (!a.length) return 0;
  const mid = Math.floor(a.length / 2);
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
}

function normalizeRepeatLine(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/[0-9]/g, '0')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

async function visionDocumentOcr(pngBuffer) {
  if (GOOGLE_ADC_ENV_ERROR) {
    return { dump: null, error: GOOGLE_ADC_ENV_ERROR };
  }
  // Avoid triggering google auth machinery when we know creds aren't configured.
  // (Railway won't have metadata-based ADC unless explicitly configured.)
  if (!process.env.GOOGLE_APPLICATION_CREDENTIALS && !process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON) {
    return { dump: null, error: 'google_vision_no_credentials_configured' };
  }
  try {
    const client = new ImageAnnotatorClient();
    const [result] = await client.documentTextDetection({ image: { content: pngBuffer } });
    return { dump: result, error: null };
  } catch (e) {
    return { dump: null, error: `google_vision_call_error:${e?.message || String(e)}` };
  }
}

function visionDumpToTokens(dump) {
  const tokens = [];
  const pages = dump?.fullTextAnnotation?.pages || [];
  for (const p of pages) {
    for (const block of p.blocks || []) {
      for (const para of block.paragraphs || []) {
        for (const word of para.words || []) {
          const text = (word.symbols || []).map((s) => s.text || '').join('');
          const bbox = bboxFromVertices(word.boundingBox?.vertices || []);
          if (!text) continue;
          tokens.push({
            text,
            bbox,
            cx: (bbox.x0 + bbox.x1) / 2,
            cy: (bbox.y0 + bbox.y1) / 2,
            h: Math.max(1, bbox.y1 - bbox.y0),
            w: Math.max(1, bbox.x1 - bbox.x0),
          });
        }
      }
    }
  }
  return tokens;
}

function tokensToLines(tokens) {
  const heights = tokens.map((t) => t.h);
  const medH = median(heights) || 10;
  const yTol = Math.max(3, medH * 0.7);
  const sorted = tokens.slice().sort((a, b) => a.cy - b.cy || a.cx - b.cx);
  const lines = [];
  let current = null;
  for (const t of sorted) {
    if (!current || Math.abs(t.cy - current.cy) > yTol) {
      current = { cy: t.cy, tokens: [t] };
      lines.push(current);
    } else {
      // Update running center to smooth OCR jitter.
      current.cy = (current.cy * current.tokens.length + t.cy) / (current.tokens.length + 1);
      current.tokens.push(t);
    }
  }
  return lines.map((l) => {
    const toks = l.tokens.slice().sort((a, b) => a.cx - b.cx);
    const text = toks.map((t) => t.text).join(' ').replace(/\s+/g, ' ').trim();
    let bb = null;
    for (const t of toks) bb = bboxUnion(bb, t.bbox);
    return { text, bbox: bb, cy: l.cy, tokens: toks };
  });
}

function detectTableRegion(lines, imageW, imageH) {
  const dateRe = /\b(\d{1,2}\s+[A-Za-z]{3}\s+\d{2,4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b/;
  let header = null;
  let headerScore = 0;
  for (const ln of lines) {
    const t = ln.text.toLowerCase();
    let score = 0;
    if (t.includes('date')) score += 2;
    if (t.includes('payment') || t.includes('type')) score += 1;
    if (t.includes('paid') || t.includes('money out') || t.includes('money in')) score += 1;
    if (t.includes('balance')) score += 1;
    if (score > headerScore) {
      headerScore = score;
      header = ln;
    }
  }

  // Find likely last row by scanning for date-ish lines + amounts.
  let lastRow = null;
  for (const ln of lines) {
    if (!ln.bbox) continue;
    if (header && ln.bbox.y0 < header.bbox.y0) continue;
    const hasDate = dateRe.test(ln.text);
    const hasAmount = /\b\d+\.\d{2}\b/.test(ln.text);
    if (hasDate || hasAmount) lastRow = ln;
  }

  if (header?.bbox && lastRow?.bbox) {
    const x0 = 0;
    const x1 = imageW;
    const y0 = clamp(header.bbox.y0 - 4, 0, imageH);
    const y1 = clamp(lastRow.bbox.y1 + 6, 0, imageH);
    return { x0, y0, x1, y1, headerScore };
  }

  // Fallback: crop the middle-lower area if we can't find header/rows.
  return {
    x0: 0,
    y0: Math.floor(imageH * 0.2),
    x1: imageW,
    y1: Math.floor(imageH * 0.95),
    headerScore,
  };
}

async function cropPng(pngBuffer, crop) {
  const img = await loadImage(pngBuffer);
  const x0 = clamp(Math.floor(crop.x0), 0, img.width);
  const y0 = clamp(Math.floor(crop.y0), 0, img.height);
  const x1 = clamp(Math.ceil(crop.x1), 0, img.width);
  const y1 = clamp(Math.ceil(crop.y1), 0, img.height);
  const w = Math.max(1, x1 - x0);
  const h = Math.max(1, y1 - y0);
  const canvas = createCanvas(w, h);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, x0, y0, w, h, 0, 0, w, h);
  const out = await canvas.encode('png');
  return Buffer.isBuffer(out) ? out : Buffer.from(out);
}

function sendConversionResponse(req, res, pngBuffer, meta) {
  setMetadataHeaders(res, meta);
  if (wantsJsonResponse(req)) {
    return res.json({
      ok: true,
      contentType: 'image/png',
      metadata: meta,
      imageBase64: pngBuffer.toString('base64'),
    });
  }
  res.setHeader('Content-Type', 'image/png');
  return res.send(pngBuffer);
}

function getFilename(req, fallback) {
  return req.file?.originalname || req.get('X-Filename') || fallback;
}

async function getPdfPageCount(pdfBuffer) {
  const pdfjsLib = await import('pdfjs-dist/legacy/build/pdf.mjs');
  const data = new Uint8Array(pdfBuffer);
  const doc = await pdfjsLib.getDocument({ data, useSystemFonts: true }).promise;
  const numPages = doc.numPages;
  await doc.destroy();
  return numPages;
}

async function renderPdfWithPdfjs(pdfBuffer, pageNumber, numPages, dpi) {
  const pdfjsLib = await import('pdfjs-dist/legacy/build/pdf.mjs');
  const data = new Uint8Array(pdfBuffer);
  const doc = await pdfjsLib.getDocument({ data, useSystemFonts: true }).promise;
  const page = await doc.getPage(Math.min(pageNumber, numPages || doc.numPages));
  // pdfjs uses scale (not dpi). Use a simple proportional mapping.
  const scale = Math.max(0.5, Math.min(6, Number(dpi || PDF_RENDER_DPI) / 100));
  const viewport = page.getViewport({ scale });
  const w = Math.floor(viewport.width);
  const h = Math.floor(viewport.height);
  if (w * h > PDF_MAX_PIXELS) {
    throw new Error(`pdfjs viewport too large (${w}x${h}).`);
  }

  const canvas = createCanvas(w, h);
  const ctx = canvas.getContext('2d');
  await page.render({
    canvasContext: ctx,
    viewport,
  }).promise;

  const pngData = await canvas.encode('png');
  const pngBuffer = Buffer.isBuffer(pngData) ? pngData : Buffer.from(pngData);
  await doc.destroy();
  return {
    pngBuffer,
    meta: {
      pageCount: numPages || doc.numPages,
      pageNumber: Math.min(pageNumber, numPages || doc.numPages),
      width: w,
      height: h,
      renderer: 'pdfjs',
    },
  };
}

async function renderPdfWithPdftocairo(pdfBuffer, pageNumber, numPages, dpi) {
  let tmpDir = null;
  try {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pdf-poppler-'));
    const inputPath = path.join(tmpDir, 'input.pdf');
    const outputPrefix = path.join(tmpDir, 'page');
    const outputPath = `${outputPrefix}.png`;
    fs.writeFileSync(inputPath, pdfBuffer);

    execFileSync(
      'pdftocairo',
      [
        '-png',
        '-singlefile',
        '-f', String(pageNumber),
        '-l', String(pageNumber),
        '-r', String(Number(dpi || PDF_RENDER_DPI)),
        inputPath,
        outputPrefix,
      ],
      { stdio: 'pipe', timeout: PDF_RENDER_TIMEOUT_MS }
    );

    if (!fs.existsSync(outputPath)) {
      throw new Error('pdftocairo produced no PNG output.');
    }
    const pngBuffer = fs.readFileSync(outputPath);
    const img = await loadImage(pngBuffer);
    const w = img.width || 0;
    const h = img.height || 0;
    if (w * h > PDF_MAX_PIXELS) {
      throw new Error(`pdftocairo output too large (${w}x${h}).`);
    }
    return {
      pngBuffer,
      meta: {
        pageCount: numPages,
        pageNumber,
        width: w,
        height: h,
        renderer: 'pdftocairo',
      },
    };
  } finally {
    if (tmpDir && fs.existsSync(tmpDir)) {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  }
}

async function renderPdfWithGhostscript(pdfBuffer, pageNumber, numPages, dpi) {
  let tmpDir = null;
  try {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pdf-gs-'));
    const inputPath = path.join(tmpDir, 'input.pdf');
    const outputPath = path.join(tmpDir, 'page.png');
    fs.writeFileSync(inputPath, pdfBuffer);

    execFileSync(
      'gs',
      [
        '-dSAFER',
        '-dBATCH',
        '-dNOPAUSE',
        '-sDEVICE=png16m',
        '-dTextAlphaBits=4',
        '-dGraphicsAlphaBits=4',
        `-r${Number(dpi || PDF_RENDER_DPI)}`,
        `-dFirstPage=${pageNumber}`,
        `-dLastPage=${pageNumber}`,
        `-sOutputFile=${outputPath}`,
        inputPath,
      ],
      { stdio: 'pipe', timeout: PDF_RENDER_TIMEOUT_MS }
    );

    if (!fs.existsSync(outputPath)) {
      throw new Error('ghostscript produced no PNG output.');
    }
    const pngBuffer = fs.readFileSync(outputPath);
    const img = await loadImage(pngBuffer);
    const w = img.width || 0;
    const h = img.height || 0;
    if (w * h > PDF_MAX_PIXELS) {
      throw new Error(`ghostscript output too large (${w}x${h}).`);
    }
    return {
      pngBuffer,
      meta: {
        pageCount: numPages,
        pageNumber,
        width: w,
        height: h,
        renderer: 'ghostscript',
      },
    };
  } finally {
    if (tmpDir && fs.existsSync(tmpDir)) {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  }
}

async function pdfToPng(pdfBuffer, pageNumber = 1, opts = {}) {
  let numPages = null;
  try {
    numPages = await getPdfPageCount(pdfBuffer);
  } catch (e) {
    // If page-count probing fails, still try renderers with requested page.
    console.warn('Could not read PDF page count:', e.message);
  }
  const safePage = numPages ? Math.min(pageNumber, numPages) : pageNumber;
  const attempts = [];

  const dpi = Number(opts.dpi || PDF_RENDER_DPI);
  for (const renderer of PDF_RENDERERS) {
    try {
      if (renderer === 'pdftocairo') {
        return await renderPdfWithPdftocairo(pdfBuffer, safePage, numPages, dpi);
      }
      if (renderer === 'ghostscript') {
        return await renderPdfWithGhostscript(pdfBuffer, safePage, numPages, dpi);
      }
      if (renderer === 'pdfjs') {
        return await renderPdfWithPdfjs(pdfBuffer, safePage, numPages, dpi);
      }
      attempts.push({ renderer, error: 'unknown renderer' });
    } catch (e) {
      attempts.push({ renderer, error: e.message });
    }
  }
  throw new Error(`All renderers failed: ${JSON.stringify(attempts)}`);
}

async function handlePdfRequest(req, res) {
  const buf = getInputBuffer(req);
  if (!buf || buf.length === 0) {
    return res.status(400).json({ error: 'No file provided. Send multipart file or raw body.' });
  }
  if (buf.length > MAX_BYTES) {
    return res.status(413).json({ error: `File exceeds max size (${MAX_MB}MB).` });
  }
  const filename = getFilename(req, 'document.pdf');
  if (!checkAllowedType(filename)) {
    return res.status(400).json({ error: 'File type not allowed.', allowedTypes: ALLOWED_TYPES });
  }

  const pageNumber = Math.max(1, parseInt(req.query.page || '1', 10));
  const dpi = Number(req.query.dpi || req.get('X-PDF-Render-DPI') || PDF_RENDER_DPI);
  const { pngBuffer, meta } = await pdfToPng(buf, pageNumber, { dpi });
  return sendConversionResponse(req, res, pngBuffer, meta);
}

async function handleOfficeRequest(req, res) {
  let tmpDir = null;
  try {
    const buf = getInputBuffer(req);
    if (!buf || buf.length === 0) {
      return res.status(400).json({ error: 'No file provided. Send multipart file or raw body.' });
    }
    if (buf.length > MAX_BYTES) {
      return res.status(413).json({ error: `File exceeds max size (${MAX_MB}MB).` });
    }
    const filename = getFilename(req, 'document.docx');
    if (!checkAllowedType(filename, true)) {
      return res.status(400).json({ error: 'File type not allowed.', allowedTypes: ALLOWED_TYPES });
    }

    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'office-'));
    const ext = path.extname(filename) || '.docx';
    const inputPath = path.join(tmpDir, `input${ext}`);
    fs.writeFileSync(inputPath, buf);

    execSync(
      `soffice --headless --convert-to pdf --outdir "${tmpDir}" "${inputPath}"`,
      { stdio: 'pipe', timeout: OFFICE_TIMEOUT_MS }
    );

    const base = path.basename(inputPath, ext);
    const pdfPath = path.join(tmpDir, `${base}.pdf`);
    if (!fs.existsSync(pdfPath)) {
      return res.status(500).json({ error: 'LibreOffice conversion produced no PDF.' });
    }
    const pdfBuffer = fs.readFileSync(pdfPath);
    const pageNumber = Math.max(1, parseInt(req.query.page || '1', 10));
    const dpi = Number(req.query.dpi || req.get('X-PDF-Render-DPI') || PDF_RENDER_DPI);
    const { pngBuffer, meta } = await pdfToPng(pdfBuffer, pageNumber, { dpi });
    return sendConversionResponse(req, res, pngBuffer, meta);
  } finally {
    if (tmpDir && fs.existsSync(tmpDir)) {
      try {
        fs.rmSync(tmpDir, { recursive: true });
      } catch (e) {
        console.warn('Temp cleanup failed:', e);
      }
    }
  }
}

app.post(
  '/convert/pdf',
  upload.single('file'),
  rawParser,
  async (req, res) => {
    try {
      await handlePdfRequest(req, res);
    } catch (err) {
      console.error('PDF convert error:', err);
      res.status(500).json({ error: 'PDF conversion failed', detail: err.message });
    }
  }
);

app.post(
  '/convert/office',
  upload.single('file'),
  rawParser,
  async (req, res) => {
    try {
      await handleOfficeRequest(req, res);
    } catch (err) {
      console.error('Office convert error:', err);
      res.status(500).json({ error: 'Office conversion failed', detail: err.message });
    }
  }
);

// Multi-page PDF -> JSON { pages: [{pageIndex, pngBase64, metadata}] }
app.post(
  '/convert-multipage',
  upload.single('file'),
  rawParser,
  async (req, res) => {
    try {
      const buf = getInputBuffer(req);
      if (!buf || buf.length === 0) {
        return res.status(400).json({ ok: false, error: 'No file provided. Send multipart file or raw body.' });
      }
      if (buf.length > MAX_BYTES) {
        return res.status(413).json({ ok: false, error: `File exceeds max size (${MAX_MB}MB).` });
      }
      const maxPages = Math.max(1, parseInt(req.query.maxPages || '50', 10));
      const dpi = Number(req.query.dpi || req.get('X-PDF-Render-DPI') || PDF_RENDER_DPI);
      const pageCount = await getPdfPageCount(buf);
      const pagesToRender = Math.min(pageCount, maxPages);
      const pages = [];
      for (let i = 1; i <= pagesToRender; i++) {
        const { pngBuffer, meta } = await pdfToPng(buf, i, { dpi });
        pages.push({
          pageIndex: i,
          metadata: meta,
          pngBase64: pngBuffer.toString('base64'),
        });
      }
      return res.json({ ok: true, pageCount, pagesRendered: pagesToRender, pages });
    } catch (err) {
      console.error('convert-multipage error:', err);
      res.status(500).json({ ok: false, error: 'Multi-page conversion failed', detail: err.message });
    }
  }
);

// End-to-end statement helper: PDF -> rendered pages -> Vision OCR -> table crop (+ debug metadata).
app.post(
  '/bank-statement/pages-to-crops',
  upload.single('file'),
  rawParser,
  async (req, res) => {
    try {
      const buf = getInputBuffer(req);
      if (!buf || buf.length === 0) {
        return res.status(400).json({ ok: false, error: 'No file provided. Send multipart file or raw body.' });
      }
      if (buf.length > MAX_BYTES) {
        return res.status(413).json({ ok: false, error: `File exceeds max size (${MAX_MB}MB).` });
      }

      const maxPages = Math.max(1, parseInt(req.query.maxPages || '20', 10));
      const dpi = Number(req.query.dpi || req.get('X-PDF-Render-DPI') || PDF_RENDER_DPI);
      const debug = String(req.query.debug || '').toLowerCase() === '1' || String(req.query.debug || '').toLowerCase() === 'true';
      const includeVisionDump = debug && (String(req.query.includeVisionDump || '').toLowerCase() !== '0');

      const pageCount = await getPdfPageCount(buf);
      const pagesToRender = Math.min(pageCount, maxPages);

      // Pass 1: render + OCR + extract lines and candidate header/footer repeat lines.
      const pageWork = [];
      const headerCounts = new Map();
      const footerCounts = new Map();
      for (let i = 1; i <= pagesToRender; i++) {
        const { pngBuffer, meta } = await pdfToPng(buf, i, { dpi });
        const { dump, error } = await visionDocumentOcr(pngBuffer);
        const tokens = dump ? visionDumpToTokens(dump) : [];
        const lines = tokensToLines(tokens);

        const headerBandMaxY = (meta.height || 0) * 0.17;
        const footerBandMinY = (meta.height || 0) * 0.83;
        const headerCandidates = [];
        const footerCandidates = [];
        for (const ln of lines) {
          if (!ln.bbox) continue;
          const norm = normalizeRepeatLine(ln.text);
          if (!norm) continue;
          if (ln.bbox.y1 <= headerBandMaxY) headerCandidates.push({ norm, text: ln.text, bbox: ln.bbox });
          if (ln.bbox.y0 >= footerBandMinY) footerCandidates.push({ norm, text: ln.text, bbox: ln.bbox });
        }
        for (const c of headerCandidates) headerCounts.set(c.norm, (headerCounts.get(c.norm) || 0) + 1);
        for (const c of footerCandidates) footerCounts.set(c.norm, (footerCounts.get(c.norm) || 0) + 1);

        pageWork.push({
          pageIndex: i,
          meta,
          pngBuffer,
          visionError: error,
          visionDump: includeVisionDump ? dump : null,
          tokensCount: tokens.length,
          lines,
          headerCandidates,
          footerCandidates,
        });
      }

      // Decide repeated header/footer norms.
      const threshold = Math.ceil(pagesToRender * 0.6);
      function pickBest(counts) {
        let best = null;
        for (const [k, v] of counts.entries()) {
          if (v < threshold) continue;
          if (!best || v > best.count) best = { norm: k, count: v };
        }
        return best;
      }
      const repeatedHeader = pickBest(headerCounts);
      const repeatedFooter = pickBest(footerCounts);

      // Pass 2: compute crop + clamp with exclusions + return images.
      const pages = [];
      for (const p of pageWork) {
        const { width, height } = p.meta;
        const lines = p.lines || [];
        const tableRegion = detectTableRegion(lines, width || 0, height || 0);

        const excludedRegions = [];
        let headerUnion = null;
        let footerUnion = null;
        if (repeatedHeader?.norm) {
          for (const c of p.headerCandidates) {
            if (c.norm === repeatedHeader.norm) headerUnion = bboxUnion(headerUnion, c.bbox);
          }
          if (headerUnion) excludedRegions.push({ kind: 'header', bbox: headerUnion, norm: repeatedHeader.norm });
        }
        if (repeatedFooter?.norm) {
          for (const c of p.footerCandidates) {
            if (c.norm === repeatedFooter.norm) footerUnion = bboxUnion(footerUnion, c.bbox);
          }
          if (footerUnion) excludedRegions.push({ kind: 'footer', bbox: footerUnion, norm: repeatedFooter.norm });
        }

        let crop = { ...tableRegion };
        if (headerUnion) crop.y0 = Math.max(crop.y0, headerUnion.y1 + 1);
        if (footerUnion) crop.y1 = Math.min(crop.y1, footerUnion.y0 - 1);
        crop.y0 = clamp(crop.y0, 0, height || crop.y0);
        crop.y1 = clamp(crop.y1, 0, height || crop.y1);

        const cropPngBuffer = await cropPng(p.pngBuffer, crop);

        pages.push({
          pageIndex: p.pageIndex,
          render: {
            contentType: 'image/png',
            pngBase64: p.pngBuffer.toString('base64'),
            metadata: p.meta,
          },
          tableRegion,
          tableCropRegion: crop,
          excludedRegions,
          tableCrop: {
            contentType: 'image/png',
            pngBase64: cropPngBuffer.toString('base64'),
          },
          debug: debug
            ? {
                visionError: p.visionError,
                tokensCount: p.tokensCount,
                headerScore: tableRegion.headerScore,
                googleVisionDump: p.visionDump,
              }
            : undefined,
        });
      }

      return res.json({
        ok: true,
        pageCount,
        pagesRendered: pagesToRender,
        repeatedHeader,
        repeatedFooter,
        pages,
      });
    } catch (err) {
      console.error('pages-to-crops error:', err);
      res.status(500).json({ ok: false, error: 'pages-to-crops failed', detail: err.message });
    }
  }
);

app.post(
  '/convert',
  upload.single('file'),
  rawParser,
  async (req, res) => {
    try {
      const filename = getFilename(req, '').toLowerCase();
      const mode = String(req.query.mode || req.get('X-Convert-Mode') || '').toLowerCase();
      const officeExtensions = new Set([
        '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods', '.odp', '.rtf'
      ]);

      const isOfficeByMode = mode === 'office';
      const isPdfByMode = mode === 'pdf';
      const isOfficeByExt = officeExtensions.has(path.extname(filename));

      if (isPdfByMode) {
        await handlePdfRequest(req, res);
        return;
      }
      if (isOfficeByMode || isOfficeByExt) {
        await handleOfficeRequest(req, res);
        return;
      }
      await handlePdfRequest(req, res);
    } catch (err) {
      console.error('Convert error:', err);
      res.status(500).json({ error: 'Conversion failed', detail: err.message });
    }
  }
);

app.get('/health', (req, res) => {
  res.json({ ok: true, maxMb: MAX_MB });
});

app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') {
    return res.status(413).json({ error: `File exceeds max size (${MAX_MB}MB).` });
  }
  console.error(err);
  res.status(500).json({ error: 'Server error' });
});

app.listen(PORT, () => {
  console.log(
    `Convert server listening on port ${PORT} (MAX_MB=${MAX_MB}, PDF_RENDERERS=${PDF_RENDERERS.join(',')})`
  );
});
