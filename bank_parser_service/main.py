import io
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber
from fastapi import FastAPI, File, Header, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel


app = FastAPI(title="TaxPig Bank Statement Parser", version="1.0.0")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",
}
ALLOWED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg"}
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
EXPECTED_BEARER = os.getenv("BANK_PARSER_BEARER_TOKEN") or os.getenv(
    "RAILWAY_BANK_PARSER_BEARER_TOKEN"
)
OCR_DPI = int(os.getenv("OCR_DPI", "250"))
MIN_WORDS_PER_PAGE = int(os.getenv("MIN_WORDS_PER_PAGE", "25"))
MAX_ROW_VERTICAL_GAP = float(os.getenv("MAX_ROW_VERTICAL_GAP", "18"))

DATE_REGEX = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})\b"
)
AMOUNT_REGEX = re.compile(
    r"(?:£|\$|€)?\s*[-(]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})\)?(?:\s*(?:CR|DR))?"
)
ROW_REGEX = r"(?P<date>\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}).*?(?P<amount>[-(]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})\)?)"


class Step(BaseModel):
    service: str
    status: str
    detail: str


class Transaction(BaseModel):
    id: str
    type: str
    date: str
    payeePayer: str
    description: str
    reference: str
    amount: float
    category: str = "Bank statement import"


class RegexPlan(BaseModel):
    row_regex: str
    stop_regex: Optional[str] = None
    notes: Optional[str] = None


def add_step(steps: List[Dict[str, str]], service: str, status: str, detail: str) -> None:
    steps.append({"service": service, "status": status, "detail": detail})


def build_error_response(
    message: str,
    steps: List[Dict[str, str]],
    notes: List[str],
    status_code: int = 400,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "text": "",
            "textPreview": "",
            "transactions": [],
            "regexPlan": None,
            "steps": steps,
            "notes": notes + [message],
        },
    )


def parse_date(value: str) -> Optional[str]:
    if not value:
        return None
    normalized = " ".join(value.strip().split())
    fmts = [
        "%d/%m/%Y",
        "%d/%m/%y",
        "%d-%m-%Y",
        "%d-%m-%y",
        "%d %b %Y",
        "%d %b %y",
        "%d %B %Y",
        "%d %B %y",
        "%b %d %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%B %d, %Y",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(normalized, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_amount(value: str) -> Optional[float]:
    if not value:
        return None
    s = value.replace(",", "").strip()
    s = s.replace("£", "").replace("$", "").replace("€", "")
    is_negative = s.startswith("(") or s.startswith("-") or s.endswith("DR") or s.endswith("-")
    s = s.replace("(", "").replace(")", "").replace("CR", "").replace("DR", "").strip()
    if s.endswith("-"):
        s = s[:-1].strip()
    try:
        val = float(s)
        return -abs(val) if is_negative else val
    except ValueError:
        return None


def extract_pdfplumber_pages(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(
                x_tolerance=2,
                y_tolerance=2,
                keep_blank_chars=False,
                use_text_flow=True,
            ) or []
            tables = page.extract_tables() or []
            text = page.extract_text() or ""
            pages.append(
                {
                    "pageNumber": idx,
                    "width": float(page.width or 0),
                    "height": float(page.height or 0),
                    "words": words,
                    "tables": tables,
                    "text": text,
                }
            )
    return pages


def quality_assess_page(page_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    words = page_data.get("words", [])
    width = float(page_data.get("width", 0))
    height = float(page_data.get("height", 0))
    text = page_data.get("text", "")
    tables = page_data.get("tables", [])

    if len(words) < MIN_WORDS_PER_PAGE:
        reasons.append(f"low_word_count:{len(words)}")

    bad_boxes = 0
    for w in words:
        x0, x1, top, bottom = w.get("x0"), w.get("x1"), w.get("top"), w.get("bottom")
        if x0 is None or x1 is None or top is None or bottom is None:
            bad_boxes += 1
            continue
        if x1 <= x0 or bottom <= top:
            bad_boxes += 1
            continue
        if x0 < -5 or x1 > width + 5 or top < -5 or bottom > height + 5:
            bad_boxes += 1

    if words and bad_boxes / max(1, len(words)) > 0.2:
        reasons.append(f"bad_box_ratio:{bad_boxes}/{len(words)}")

    if len(text.strip()) < 40 and len(words) < 40:
        reasons.append("text_too_sparse")

    malformed_tables = 0
    for tbl in tables:
        if not tbl:
            malformed_tables += 1
            continue
        max_cols = max((len(r) for r in tbl if r), default=0)
        if max_cols < 2:
            malformed_tables += 1
    if tables and malformed_tables == len(tables):
        reasons.append("malformed_table_region")

    return len(reasons) == 0, reasons


def _group_words_into_lines(words: List[Dict[str, Any]], y_tol: float = 3.0) -> List[List[Dict[str, Any]]]:
    lines: List[List[Dict[str, Any]]] = []
    sorted_words = sorted(words, key=lambda w: (float(w.get("top", 0)), float(w.get("x0", 0))))
    for word in sorted_words:
        top = float(word.get("top", 0))
        if not lines:
            lines.append([word])
            continue
        last_line_top = float(lines[-1][0].get("top", 0))
        if abs(top - last_line_top) <= y_tol:
            lines[-1].append(word)
        else:
            lines.append([word])
    return lines


def _line_text(line_words: List[Dict[str, Any]]) -> str:
    return " ".join(
        str(w.get("text", "")).strip() for w in sorted(line_words, key=lambda w: float(w.get("x0", 0)))
        if str(w.get("text", "")).strip()
    )


def _line_top(line_words: List[Dict[str, Any]]) -> float:
    return min(float(w.get("top", 0)) for w in line_words) if line_words else 0.0


def _is_probable_header(line_text: str) -> bool:
    t = line_text.lower()
    header_tokens = [
        "balance",
        "statement",
        "page",
        "sort code",
        "account number",
        "transactions",
        "description",
        "reference",
        "debit",
        "credit",
    ]
    return any(tok in t for tok in header_tokens) and not DATE_REGEX.search(line_text)


def _is_summary_non_transaction_line(line_text: str) -> bool:
    t = line_text.lower()
    if t.startswith("opening balance") or t.startswith("closing balance"):
        return True
    if t.startswith("balance on "):
        return True
    if t.startswith("money in") or t.startswith("money out"):
        return True
    if "statement date" in t or "sort code" in t or "account number" in t:
        return True
    return False


def _choose_primary_amount(amount_matches: List[str], line_text: str) -> str:
    if not amount_matches:
        return ""
    cleaned: List[str] = []
    for m in amount_matches:
        s = " ".join(str(m).split())
        if s and s not in cleaned:
            cleaned.append(s)
    if not cleaned:
        return ""

    signed = [x for x in cleaned if "-" in x or "(" in x or "DR" in x.upper()]
    if signed:
        return signed[0]

    if "balance" in line_text.lower() and len(cleaned) >= 2:
        scored: List[Tuple[float, str]] = []
        for token in cleaned:
            parsed = parse_amount(token)
            if parsed is not None:
                scored.append((abs(parsed), token))
        if scored:
            scored.sort(key=lambda x: x[0])
            return scored[0][1]

    # Common bank layout: transaction amount appears before running balance.
    if len(cleaned) >= 2:
        return cleaned[0]
    return cleaned[0]


def reconstruct_rows_geometric(page_data: Dict[str, Any], source: str = "pdfplumber") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    page_number = page_data.get("pageNumber", 0)
    width = float(page_data.get("width", 0))
    words = page_data.get("words", [])
    tables = page_data.get("tables", [])

    # Table-first candidates.
    for table in tables:
        for row in table:
            if not row:
                continue
            values = [((c or "").strip()) for c in row]
            joined = " ".join(v for v in values if v)
            if not joined:
                continue
            date_match = DATE_REGEX.search(joined)
            amount_matches = AMOUNT_REGEX.findall(joined)
            if date_match and amount_matches:
                if _is_summary_non_transaction_line(joined):
                    continue
                rows.append(
                    {
                        "source": source,
                        "pageNumber": page_number,
                        "dateRaw": date_match.group(1),
                        "amountRaw": _choose_primary_amount(amount_matches, joined),
                        "descriptionRaw": joined,
                        "referenceRaw": "",
                    }
                )

    # Geometric line reconstruction from words (layout-agnostic).
    lines = _group_words_into_lines(words)
    current_row: Optional[Dict[str, Any]] = None
    pending_date_row: Optional[Dict[str, Any]] = None
    pending_date_top: float = -1.0

    for line in lines:
        line_sorted = sorted(line, key=lambda w: float(w.get("x0", 0)))
        line_text = _line_text(line_sorted)
        if not line_text:
            continue
        if _is_probable_header(line_text):
            continue

        date_match = DATE_REGEX.search(line_text)
        right_side = [w for w in line_sorted if float(w.get("x0", 0)) > (width * 0.62)]
        right_text = _line_text(right_side)
        line_amounts = AMOUNT_REGEX.findall(line_text)
        right_amounts = AMOUNT_REGEX.findall(right_text)
        if "balance" in line_text.lower() and len(line_amounts) >= 2:
            amount_matches = line_amounts
        else:
            amount_matches = right_amounts or line_amounts
        line_top = _line_top(line_sorted)

        if date_match and amount_matches:
            if _is_summary_non_transaction_line(line_text):
                continue
            if current_row:
                rows.append(current_row)
            current_row = {
                "source": source,
                "pageNumber": page_number,
                "dateRaw": date_match.group(1),
                "amountRaw": _choose_primary_amount(amount_matches, line_text),
                "descriptionRaw": line_text,
                "referenceRaw": "",
            }
            pending_date_row = None
            continue

        # Date in one line/column, amount in another nearby line/column.
        if date_match and not amount_matches:
            pending_date_row = {
                "source": source,
                "pageNumber": page_number,
                "dateRaw": date_match.group(1),
                "amountRaw": "",
                "descriptionRaw": line_text,
                "referenceRaw": "",
            }
            pending_date_top = line_top
            if current_row:
                rows.append(current_row)
                current_row = None
            continue

        if amount_matches and pending_date_row:
            if abs(line_top - pending_date_top) <= MAX_ROW_VERTICAL_GAP:
                combined = dict(pending_date_row)
                combined["amountRaw"] = amount_matches[-1]
                combined["descriptionRaw"] = f"{pending_date_row['descriptionRaw']} {line_text}".strip()
                rows.append(combined)
                pending_date_row = None
                pending_date_top = -1.0
                continue

        # Continuation line for the active row.
        if current_row and not date_match and not amount_matches:
            current_row["descriptionRaw"] = f"{current_row['descriptionRaw']} {line_text}".strip()

    if current_row:
        rows.append(current_row)

    # Deduplicate by stable composite key.
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        k = (
            row.get("pageNumber"),
            (row.get("dateRaw") or "").strip(),
            (row.get("amountRaw") or "").strip(),
            " ".join((row.get("descriptionRaw") or "").split())[:120].lower(),
        )
        if k in seen:
            continue
        seen.add(k)
        deduped.append(row)

    return deduped


def ocr_page_if_needed(pdf_bytes: bytes, page_number: int, reasons: List[str]) -> Tuple[List[Dict[str, Any]], str]:
    try:
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
            from pytesseract import Output
        except Exception as exc:
            return [], f"ocr_unavailable:{exc}"

        images = convert_from_bytes(
            pdf_bytes,
            dpi=OCR_DPI,
            first_page=page_number,
            last_page=page_number,
            fmt="png",
        )
        if not images:
            return [], "ocr_no_image"

        ocr = pytesseract.image_to_data(images[0], output_type=Output.DICT)
        words: List[Dict[str, Any]] = []
        extracted_words = 0
        n = len(ocr.get("text", []))
        for i in range(n):
            txt = (ocr.get("text", [""])[i] or "").strip()
            conf = float(ocr.get("conf", ["-1"])[i] or -1)
            if not txt or conf < 35:
                continue
            left = float(ocr.get("left", [0])[i])
            top = float(ocr.get("top", [0])[i])
            w = float(ocr.get("width", [0])[i])
            h = float(ocr.get("height", [0])[i])
            words.append(
                {
                    "text": txt,
                    "x0": left,
                    "x1": left + w,
                    "top": top,
                    "bottom": top + h,
                }
            )
            extracted_words += 1

        page_data = {
            "pageNumber": page_number,
            "width": float(images[0].width),
            "height": float(images[0].height),
            "words": words,
            "tables": [],
            "text": " ".join(w["text"] for w in words),
        }
        return reconstruct_rows_geometric(page_data, source="ocr"), (
            f"ocr_applied:page={page_number},words={extracted_words},reasons={','.join(reasons)}"
        )
    except Exception as exc:
        return [], f"ocr_failed:{exc}"


def ocr_image_file(image_bytes: bytes) -> Tuple[List[Dict[str, Any]], str]:
    try:
        import pytesseract
        from pytesseract import Output
        from PIL import Image
    except Exception as exc:
        return [], f"ocr_unavailable:{exc}"

    try:
        img = Image.open(io.BytesIO(image_bytes))
        ocr = pytesseract.image_to_data(img, output_type=Output.DICT)
        words: List[Dict[str, Any]] = []
        n = len(ocr.get("text", []))
        extracted_words = 0
        for i in range(n):
            txt = (ocr.get("text", [""])[i] or "").strip()
            conf = float(ocr.get("conf", ["-1"])[i] or -1)
            if not txt or conf < 35:
                continue
            left = float(ocr.get("left", [0])[i])
            top = float(ocr.get("top", [0])[i])
            w = float(ocr.get("width", [0])[i])
            h = float(ocr.get("height", [0])[i])
            words.append({"text": txt, "x0": left, "x1": left + w, "top": top, "bottom": top + h})
            extracted_words += 1
        page_data = {
            "pageNumber": 1,
            "width": float(getattr(img, "width", 0)),
            "height": float(getattr(img, "height", 0)),
            "words": words,
            "tables": [],
            "text": " ".join(w["text"] for w in words),
        }
        return reconstruct_rows_geometric(page_data, source="ocr_image"), f"ocr_image_words={extracted_words}"
    except Exception as exc:
        return [], f"ocr_image_failed:{exc}"


def repair_rows_with_gpt_compact(
    headers: List[str],
    sample_rows: List[Dict[str, Any]],
    failed_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str]:
    # Intentionally conservative: tiny-call policy compliant stub.
    # No full OCR dumps are sent; this function can be upgraded to call GPT with
    # only compact slices (headers, sample rows, failed rows).
    _ = headers, sample_rows, failed_rows
    return [], "gpt_skipped_compact_policy_no_client"


def normalize_transactions(raw_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    transactions: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for row in raw_rows:
        date_value = parse_date(str(row.get("dateRaw", "")).strip())
        amount_value = parse_amount(str(row.get("amountRaw", "")).strip())
        description = " ".join(str(row.get("descriptionRaw", "")).split())
        if _is_summary_non_transaction_line(description):
            failed.append(row)
            continue
        if not date_value or amount_value is None or not description:
            failed.append(row)
            continue
        reference = ""
        ref_match = re.search(r"\b(ref(?:erence)?[:\s-]*[A-Z0-9-]{4,})\b", description, re.I)
        if ref_match:
            reference = ref_match.group(1).strip()
        d = description.lower()
        if amount_value < 0:
            tx_type = "expense"
        elif re.search(r"\bdeb(it)?\b|\bdr\b|money out|withdrawal|card|purchase|payment to|transfer out", d):
            tx_type = "expense"
        elif re.search(r"\bcr\b|credit|money in|salary|refund|interest|reward|transfer in", d):
            tx_type = "income"
        else:
            tx_type = "income"
        tx = {
            "id": str(uuid.uuid4()),
            "type": tx_type,
            "date": date_value,
            "payeePayer": description[:120],
            "description": description,
            "reference": reference,
            "amount": abs(round(float(amount_value), 2)),
            "category": "Bank statement import",
        }
        transactions.append(tx)
    return transactions, failed


@app.post("/parse-bank-statement")
async def parse_bank_statement(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
):
    steps: List[Dict[str, str]] = []
    notes: List[str] = []
    reconstructed_text_chunks: List[str] = []

    try:
        if EXPECTED_BEARER:
            if not authorization or authorization != f"Bearer {EXPECTED_BEARER}":
                add_step(steps, "auth", "failed", "Bearer token missing or invalid.")
                return build_error_response("Unauthorized", steps, notes, status_code=401)
            add_step(steps, "auth", "success", "Bearer token accepted.")
        else:
            add_step(steps, "auth", "skipped", "No bearer token configured in env.")

        filename = file.filename or "upload.pdf"
        ext = os.path.splitext(filename)[1].lower()
        mime = (file.content_type or "").lower()
        if ext not in ALLOWED_EXTENSIONS:
            add_step(steps, "validation", "failed", f"Invalid extension: {ext}")
            return build_error_response("Only PDF/PNG/JPG files are accepted.", steps, notes, status_code=400)
        allowed_mimes = ALLOWED_MIME_TYPES.union(ALLOWED_IMAGE_MIME_TYPES)
        if mime and mime not in allowed_mimes:
            add_step(steps, "validation", "failed", f"Invalid content type: {mime}")
            return build_error_response("Unsupported content type.", steps, notes, status_code=400)

        pdf_bytes = await file.read()
        if not pdf_bytes:
            add_step(steps, "validation", "failed", "Empty upload payload.")
            return build_error_response("Empty upload.", steps, notes, status_code=400)
        if len(pdf_bytes) > MAX_UPLOAD_BYTES:
            add_step(steps, "validation", "failed", f"File exceeds {MAX_UPLOAD_MB}MB.")
            return build_error_response(
                f"File too large (max {MAX_UPLOAD_MB}MB).", steps, notes, status_code=413
            )
        add_step(steps, "validation", "success", f"Accepted {filename} ({len(pdf_bytes)} bytes).")

        raw_rows: List[Dict[str, Any]] = []
        ocr_pages: List[str] = []
        if ext == ".pdf":
            pages = extract_pdfplumber_pages(pdf_bytes)
            add_step(steps, "pdfplumber", "success", f"Extracted {len(pages)} pages.")
            for page_data in pages:
                page_number = int(page_data["pageNumber"])
                reconstructed_text_chunks.append(page_data.get("text", ""))
                ok, reasons = quality_assess_page(page_data)
                if ok:
                    page_rows = reconstruct_rows_geometric(page_data, source="pdfplumber")
                    raw_rows.extend(page_rows)
                    add_step(
                        steps,
                        "quality_assess_page",
                        "success",
                        f"page={page_number} pdfplumber_ok rows={len(page_rows)}",
                    )
                else:
                    add_step(
                        steps,
                        "quality_assess_page",
                        "failed",
                        f"page={page_number} reasons={','.join(reasons)}",
                    )
                    ocr_rows, ocr_detail = ocr_page_if_needed(pdf_bytes, page_number, reasons)
                    if ocr_rows:
                        raw_rows.extend(ocr_rows)
                        ocr_pages.append(f"page {page_number}: {ocr_detail}")
                        add_step(steps, "ocr_page_if_needed", "success", ocr_detail)
                    else:
                        notes.append(f"OCR skipped/failed for page {page_number}: {ocr_detail}")
                        add_step(steps, "ocr_page_if_needed", "failed", ocr_detail)
        else:
            ocr_rows, ocr_detail = ocr_image_file(pdf_bytes)
            if ocr_rows:
                raw_rows.extend(ocr_rows)
                add_step(steps, "ocr_image_file", "success", ocr_detail)
            else:
                add_step(steps, "ocr_image_file", "failed", ocr_detail)
                notes.append(f"Image OCR failed: {ocr_detail}")

        add_step(steps, "reconstruct_rows_geometric", "success", f"raw_rows={len(raw_rows)}")
        transactions, failed_rows = normalize_transactions(raw_rows)
        add_step(
            steps,
            "normalize_transactions",
            "success",
            f"transactions={len(transactions)} failed_rows={len(failed_rows)}",
        )

        repaired_rows, gpt_detail = repair_rows_with_gpt_compact(
            headers=["date", "description", "amount"],
            sample_rows=raw_rows[:20],
            failed_rows=failed_rows[:20],
        )
        if repaired_rows:
            repaired_transactions, _ = normalize_transactions(repaired_rows)
            transactions.extend(repaired_transactions)
            add_step(
                steps,
                "repair_rows_with_gpt_compact",
                "success",
                f"{gpt_detail}; repaired={len(repaired_transactions)}",
            )
        else:
            add_step(steps, "repair_rows_with_gpt_compact", "skipped", gpt_detail)

        if ocr_pages:
            notes.append("OCR fallback pages: " + "; ".join(ocr_pages))
        if not transactions:
            notes.append("No transactions parsed. Statement layout may need template tuning.")

        full_text = "\n\n".join(c for c in reconstructed_text_chunks if c).strip()
        regex_plan = RegexPlan(
            row_regex=ROW_REGEX,
            stop_regex=None,
            notes="Geometric date+amount row detection with continuation-line merge.",
        )
        return {
            "ok": True,
            "text": full_text if full_text else "",
            "textPreview": full_text[:2000] if full_text else "",
            "transactions": [Transaction(**t).model_dump() for t in transactions],
            "regexPlan": regex_plan.model_dump(),
            "steps": [Step(**s).model_dump() for s in steps],
            "notes": notes,
        }
    except Exception as exc:
        add_step(steps, "parse-bank-statement", "failed", f"Unhandled error: {exc}")
        return build_error_response("Parser error", steps, notes + [str(exc)], status_code=500)

