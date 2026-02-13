# Bank Parser Service (FastAPI)

This service exposes:

- `POST /parse-bank-statement`

It accepts a single multipart file field named `file` and returns structured transactions with `steps` and `notes`.
Supported uploads: `pdf`, `png`, `jpg`, `jpeg` (PDF-first, OCR fallback).

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
- `OPENAI_API_KEY` (optional, enables compact GPT repair/validation)
- `GPT_MODEL` (optional, default `gpt-4o-mini`)
- `CONFIDENCE_REVIEW_THRESHOLD` (default `0.55`)

## Example request

```bash
curl -X POST "http://localhost:8000/parse-bank-statement" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@../invoice-233580.pdf"
```

## Notes

- Uses `pdfplumber` first for digital PDFs.
- OCR is page-level fallback only when page quality checks fail.
- Compact GPT integration sends only headers + sampled rows + failed rows (never full OCR dump).
- For image files, OCR is used directly.

## Batch test helper

Run all files in `testfiles/` and print a summary:

```bash
cd bank_parser_service
./.venv/bin/python testfiles/run_batch_eval.py
```

## View transactions as a table

### Browser viewer

Run the API locally and open:

`http://localhost:8000/parse-bank-statement/viewer`

Upload a file, and it renders the parsed transactions in an HTML table.

### Terminal table output

```bash
cd bank_parser_service
./.venv/bin/python print_transactions_table.py \
  --url "http://localhost:8000/parse-bank-statement" \
  --file "testfiles/Bank Statement to Feb 2021.pdf" \
  --token "YOUR_TOKEN"
```

## Local OCR dependencies

macOS:

```bash
brew install tesseract poppler
```
