import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import (
    extract_pdfplumber_pages,
    normalize_transactions,
    ocr_image_file,
    quality_assess_page,
    reconstruct_rows_geometric,
)


def evaluate_file(path: Path) -> dict:
    ext = path.suffix.lower()
    raw_rows = []
    page_quality = []
    payload = path.read_bytes()

    if ext == ".pdf":
        pages = extract_pdfplumber_pages(payload)
        for pg in pages:
            ok, reasons = quality_assess_page(pg)
            page_quality.append(
                {
                    "page": pg["pageNumber"],
                    "words": len(pg.get("words", [])),
                    "ok": ok,
                    "reasons": reasons,
                }
            )
            raw_rows.extend(reconstruct_rows_geometric(pg, source="pdfplumber"))
    elif ext in {".png", ".jpg", ".jpeg"}:
        rows, detail = ocr_image_file(payload)
        raw_rows.extend(rows)
        page_quality.append({"page": 1, "words": None, "ok": bool(rows), "reasons": [detail]})
    else:
        return {"file": path.name, "status": "skipped", "reason": f"unsupported extension: {ext}"}

    transactions, failed_rows = normalize_transactions(raw_rows)
    return {
        "file": path.name,
        "status": "ok",
        "rawRows": len(raw_rows),
        "transactions": len(transactions),
        "failedRows": len(failed_rows),
        "pageQuality": page_quality,
        "sampleTransactions": transactions[:5],
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    files = sorted(
        [p for p in here.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}]
    )
    results = [evaluate_file(p) for p in files]
    print(json.dumps({"count": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
