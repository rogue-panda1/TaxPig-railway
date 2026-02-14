# Railway Runtime Info

- `baseurl`: `https://taxpig-railway-production.up.railway.app`
- Service health: `GET /health`
- Convert endpoints:
  - `POST /convert`
  - `POST /convert/pdf`
  - `POST /convert/office`
  - `POST /convert-multipage` (multi-page PDF -> JSON pages[])
  - `POST /bank-statement/pages-to-crops` (PDF -> rendered pages + table crop + debug)

## Bank statement helpers

### Multi-page render (PDF -> pages[])

```bash
curl -sS -X POST "https://taxpig-railway-production.up.railway.app/convert-multipage?maxPages=10&dpi=200" \
  -H "Accept: application/json" \
  -H "Content-Type: application/pdf" \
  --data-binary @"Bank Statement to Feb 2021.pdf"
```

### OCR + crop (PDF -> crops; enable debug + Vision dump)

```bash
curl -sS -X POST "https://taxpig-railway-production.up.railway.app/bank-statement/pages-to-crops?maxPages=5&dpi=200&debug=1&includeVisionDump=1" \
  -H "Accept: application/json" \
  -H "Content-Type: application/pdf" \
  --data-binary @"Bank Statement to Feb 2021.pdf"
```

## Google Vision credentials (service account)

Set this env var on the Railway service running `node server.js` (the convert service):

- `GOOGLE_APPLICATION_CREDENTIALS_JSON`: service-account JSON (either raw JSON or base64).

The service writes it to `/tmp/google_application_credentials.json` at startup and sets
`GOOGLE_APPLICATION_CREDENTIALS` automatically.

## Quick checks

### Health

```bash
curl -s https://taxpig-railway-production.up.railway.app/health
```

### PDF conversion + renderer check

```bash
curl -s -X POST "https://taxpig-railway-production.up.railway.app/convert/pdf" \
  -F "file=@invoice-233580.pdf" \
  -o out.png -D headers.txt
grep -i "^X-Renderer:" headers.txt
```

### JSON response mode (for callers expecting JSON)

```bash
curl -s -X POST "https://taxpig-railway-production.up.railway.app/convert?response=json" \
  -F "file=@invoice-233580.pdf"
```

## Expected renderer order

Configured fallback order:

1. `pdftocairo` (Poppler)
2. `ghostscript`
3. `pdfjs`

Read `X-Renderer` response header to confirm which renderer served the output.
