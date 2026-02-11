# TaxPig Convert (Railway)

Document conversion API: **PDF → PNG** and **Office (DOCX, XLSX, etc.) → PDF → PNG**.  
Runs on [Railway](https://railway.app) with optional size and type restrictions.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/convert` | Compatibility route: auto-detect PDF vs Office by extension; supports `?mode=pdf|office` |
| POST | `/convert/pdf` | Convert PDF to PNG (first page by default) |
| POST | `/convert/office` | Convert Office document to PDF then to PNG |
| GET | `/health` | Health check |

- **Input:** multipart form (`file` field) or raw body (binary).
- **Output (default):** `image/png` body; metadata in response headers (base64 JSON in `X-Conversion-Metadata`, plus `X-Page-Count`, `X-Page-Number`, `X-Width`, `X-Height`).
- **Output (optional JSON):** add `?response=json` (or header `X-Response-Format: json`, or `Accept: application/json`) to get JSON:
  - `ok`
  - `contentType` (`image/png`)
  - `metadata`
  - `imageBase64`
- **Query:** `?page=2` for a specific page (default `1`).

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | No | `3000` | Server port (Railway sets this) |
| `MAX_MB` | No | `25` | Max upload size in MB |
| `ALLOWED_TYPES` | No | *(all)* | Comma-separated extensions, e.g. `pdf,docx,xlsx` |

## cURL examples

**PDF → PNG (multipart):**

```bash
curl -X POST http://localhost:3000/convert/pdf \
  -F "file=@document.pdf" \
  --output page.png \
  -D -
```

**Compatibility route (`/convert`) with PDF:**

```bash
curl -X POST http://localhost:3000/convert \
  -F "file=@document.pdf" \
  --output page.png \
  -D -
```

**Compatibility route (`/convert`) with Office mode forced:**

```bash
curl -X POST "http://localhost:3000/convert?mode=office" \
  -F "file=@report.docx" \
  --output page.png \
  -D -
```

**JSON response mode (for automation that expects JSON):**

```bash
curl -X POST "http://localhost:3000/convert?response=json" \
  -F "file=@document.pdf"
```

**PDF → PNG (raw buffer):**

```bash
curl -X POST http://localhost:3000/convert/pdf \
  -H "Content-Type: application/octet-stream" \
  -H "X-Filename: document.pdf" \
  --data-binary "@document.pdf" \
  --output page.png \
  -D -
```

**Specific page and show metadata headers:**

```bash
curl -X POST "http://localhost:3000/convert/pdf?page=2" \
  -F "file=@report.pdf" \
  -o page2.png \
  -v
```

**Office → PNG (multipart):**

```bash
curl -X POST http://localhost:3000/convert/office \
  -F "file=@report.docx" \
  --output page.png \
  -D -
```

**Office → PNG (raw buffer):**

```bash
curl -X POST http://localhost:3000/convert/office \
  -H "Content-Type: application/octet-stream" \
  -H "X-Filename: report.docx" \
  --data-binary "@report.docx" \
  --output page.png \
  -D -
```

**Health check:**

```bash
curl http://localhost:3000/health
```

## Reading metadata from headers

Response headers include conversion metadata:

- `X-Conversion-Metadata`: base64-encoded JSON, e.g.  
  `{"pageCount":3,"pageNumber":1,"width":1190,"height":1684}`
- `X-Page-Count`, `X-Page-Number`, `X-Width`, `X-Height`: plain values

Example (decode in shell):

```bash
curl -s -X POST http://localhost:3000/convert/pdf -F "file=@doc.pdf" -o out.png -D headers.txt
grep -i x-conversion-metadata headers.txt | cut -d' ' -f2- | base64 -d
```

## Railway setup

1. **Create a project**
   - Go to [railway.app](https://railway.app) and create a new project.
   - Choose **Deploy from GitHub repo** (or **Empty project** and connect repo later).

2. **Add this repo**
   - Connect the repo that contains this code (e.g. `TaxPig-railway`).
   - Railway will detect Node and use Nixpacks; `nixpacks.toml` installs LibreOffice for `/convert/office`.

3. **Configure (optional)**
   - In the service → **Variables**, add:
     - `MAX_MB` — e.g. `10`
     - `ALLOWED_TYPES` — e.g. `pdf,docx,xlsx,pptx`
   - `PORT` is set by Railway; no need to add it.

4. **Deploy**
   - Push to the linked branch; Railway builds and deploys.
   - Open **Settings → Networking → Generate domain** to get a public URL.

5. **Test**
   - Replace `localhost:3000` with your Railway URL:
   ```bash
   curl -X POST https://YOUR_APP.up.railway.app/convert/pdf -F "file=@test.pdf" -o out.png
   ```

## Local development

```bash
npm install
npm start
# or: npm run dev  (with --watch)
```

For `/convert/office` locally you need LibreOffice:

- **macOS:** `brew install libreoffice`
- **Ubuntu/Debian:** `sudo apt install libreoffice`

## Stack

- **PDF → PNG:** `pdfjs-dist` + `@napi-rs/canvas`
- **Office → PDF → PNG:** headless LibreOffice (`soffice`) in a temp dir, then the same PDF→PNG pipeline
