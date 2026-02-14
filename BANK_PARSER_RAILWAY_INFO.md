# TaxPig Bank Parser (Railway)

Public base URL (production):

- `https://taxpig-railway-bankstatement-production.up.railway.app`

## Endpoints

- `POST /parse-bank-statement`
  - `multipart/form-data`, field name: `file`
  - auth: `Authorization: Bearer <token>` (token comes from `RAILWAY_BANK_PARSER_BEARER_TOKEN` or `BANK_PARSER_BEARER_TOKEN`)
- `POST /parse-bank-statement/from-url?url=<https-url>`
  - optional query params:
    - `useVision=1` (Vision-first parse)
    - `includeVisionDump=1`
    - `visionMaxPages=1`
    - `visionDpi=200`
- `GET /parse-bank-statement/quick-view?url=<https-url>&token=<bearer>`
  - HTML viewer with tabs: `pdfplumber` vs `google vision`
- `GET /parse-bank-statement/vision-dump?url=<https-url>&token=<bearer>&visionMaxPages=1&visionDpi=200`
  - returns the raw Vision response JSON wrapper: `{ dpi, pages: [{ pageNumber, width, height, dump }] }`
- `GET /parse-bank-statement/viewer`
  - HTML upload viewer (posts to `/parse-bank-statement`)

## Google Vision credentials (service account)

Set this env var on the Railway *bank parser* service:

- `GOOGLE_APPLICATION_CREDENTIALS_JSON`

Value can be either:

- raw JSON (the whole service-account JSON), or
- base64 of that JSON (single line)

At startup, the service writes the JSON to:

- `/tmp/google_application_credentials.json`

and sets:

- `GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json`

## Handy curl snippets

Upload a PDF:

```bash
curl -sS -X POST "https://taxpig-railway-bankstatement-production.up.railway.app/parse-bank-statement" \
  -H "Authorization: Bearer $RAILWAY_BANK_PARSER_BEARER_TOKEN" \
  -F "file=@statement.pdf"
```

Quick HTML view (URL-based):

```bash
TOKEN="$RAILWAY_BANK_PARSER_BEARER_TOKEN"
URL="https://example.com/statement.pdf"
open "https://taxpig-railway-bankstatement-production.up.railway.app/parse-bank-statement/quick-view?url=$URL&token=$TOKEN&visionMaxPages=1&visionDpi=200"
```

