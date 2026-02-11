# Railway Runtime Info

- `baseurl`: `https://taxpig-railway-production.up.railway.app`
- Service health: `GET /health`
- Convert endpoints:
  - `POST /convert`
  - `POST /convert/pdf`
  - `POST /convert/office`

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
