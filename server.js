const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { execSync, execFileSync } = require('child_process');
const { createCanvas, loadImage } = require('@napi-rs/canvas');

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

async function renderPdfWithPdfjs(pdfBuffer, pageNumber, numPages) {
  const pdfjsLib = await import('pdfjs-dist/legacy/build/pdf.mjs');
  const data = new Uint8Array(pdfBuffer);
  const doc = await pdfjsLib.getDocument({ data, useSystemFonts: true }).promise;
  const page = await doc.getPage(Math.min(pageNumber, numPages || doc.numPages));
  const scale = 2;
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

async function renderPdfWithPdftocairo(pdfBuffer, pageNumber, numPages) {
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
        '-r', String(PDF_RENDER_DPI),
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

async function renderPdfWithGhostscript(pdfBuffer, pageNumber, numPages) {
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
        `-r${PDF_RENDER_DPI}`,
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

async function pdfToPng(pdfBuffer, pageNumber = 1) {
  let numPages = null;
  try {
    numPages = await getPdfPageCount(pdfBuffer);
  } catch (e) {
    // If page-count probing fails, still try renderers with requested page.
    console.warn('Could not read PDF page count:', e.message);
  }
  const safePage = numPages ? Math.min(pageNumber, numPages) : pageNumber;
  const attempts = [];

  for (const renderer of PDF_RENDERERS) {
    try {
      if (renderer === 'pdftocairo') {
        return await renderPdfWithPdftocairo(pdfBuffer, safePage, numPages);
      }
      if (renderer === 'ghostscript') {
        return await renderPdfWithGhostscript(pdfBuffer, safePage, numPages);
      }
      if (renderer === 'pdfjs') {
        return await renderPdfWithPdfjs(pdfBuffer, safePage, numPages);
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
  const { pngBuffer, meta } = await pdfToPng(buf, pageNumber);
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
    const { pngBuffer, meta } = await pdfToPng(pdfBuffer, pageNumber);
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
