# Bank Parser Service (FastAPI)

This service exposes:

- `POST /parse-bank-statement`

It accepts a single multipart file field named `file` and returns structured transactions with `steps` and `notes`.

## Run locally

```bash
cd bank_parser_service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Environment variables

- `BANK_PARSER_BEARER_TOKEN` (or `RAILWAY_BANK_PARSER_BEARER_TOKEN`)
- `MAX_UPLOAD_MB` (default `20`)
- `MIN_WORDS_PER_PAGE` (default `25`)
- `OCR_DPI` (default `250`)

## Example request

```bash
curl -X POST "http://localhost:8000/parse-bank-statement" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@../invoice-233580.pdf"
```

## Notes

- Uses `pdfplumber` first for digital PDFs.
- OCR is page-level fallback only when page quality checks fail.
- GPT repair function is compact-policy ready and currently conservative (skips by default).
