import io
import json
import os
import re
import uuid
import base64
from tempfile import SpooledTemporaryFile
from urllib.parse import unquote, urlparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber
from fastapi import FastAPI, File, Header, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from starlette.datastructures import Headers, UploadFile as StarletteUploadFile
import requests


app = FastAPI(title="TaxPig Bank Statement Parser", version="1.0.1")

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
GPT_MAX_SAMPLE_ROWS = int(os.getenv("GPT_MAX_SAMPLE_ROWS", "20"))
CONFIDENCE_REVIEW_THRESHOLD = float(os.getenv("CONFIDENCE_REVIEW_THRESHOLD", "0.55"))
GOOGLE_VISION_DISABLE = os.getenv("GOOGLE_VISION_DISABLE", "").strip().lower() in {"1", "true", "yes"}
GOOGLE_APPLICATION_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
# Default to "all pages" (0 = no limit) for statements.
# You can override per-request with visionMaxPages=...
GOOGLE_VISION_DEFAULT_MAX_PAGES = int(os.getenv("GOOGLE_VISION_DEFAULT_MAX_PAGES", "0"))
GOOGLE_VISION_DEFAULT_DPI = int(os.getenv("GOOGLE_VISION_DEFAULT_DPI", "200"))


def _maybe_write_google_adc_from_env() -> Optional[str]:
    """
    Railway-friendly ADC configuration.
    If GOOGLE_APPLICATION_CREDENTIALS_JSON is set, write it to /tmp and point
    GOOGLE_APPLICATION_CREDENTIALS at it.
    """
    if not GOOGLE_APPLICATION_CREDENTIALS_JSON:
        return None
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return None
    raw = GOOGLE_APPLICATION_CREDENTIALS_JSON
    try:
        if raw.lstrip().startswith("{"):
            data = json.loads(raw)
        else:
            decoded = base64.b64decode(raw.encode("utf-8")).decode("utf-8")
            data = json.loads(decoded)
    except Exception as exc:
        return f"google_adc_env_invalid:{exc}"
    try:
        path = "/tmp/google_application_credentials.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
        return None
    except Exception as exc:
        return f"google_adc_env_write_failed:{exc}"


_GOOGLE_ADC_ENV_ERROR = _maybe_write_google_adc_from_env()

DATE_REGEX = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})\b"
)
AMOUNT_REGEX = re.compile(
    r"(?:£|\$|€)?\s*[-(]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})\)?(?:\s*(?:CR|DR))?"
)
ROW_REGEX = r"(?P<date>\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}).*?(?P<amount>[-(]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})\)?)"

# Bank statement "type" codes vary by bank. Keep this broad.
PAYMENT_TYPE_REGEX = re.compile(
    r"\b(DD|VIS|BP|BGC|FPO|FPI|TFR|SO|CHQ|ATM|POS|CASH|DEB|DEP|CPT|FEE|PAY|MPI|MPO)\b",
    flags=re.I,
)


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
    transactionDescription: Optional[str] = None
    originalSourceText: Optional[str] = None
    paymentType: Optional[str] = None
    paidOutGbp: Optional[float] = None
    paidInGbp: Optional[float] = None
    balanceGbp: Optional[float] = None
    reference: str
    amount: float
    category: str = "Bank statement import"
    confidence: Optional[float] = None


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
            "statementCsv": "",
            "googleVisionDump": None,
            "googleVisionDumpError": None,
            "regexPlan": None,
            "steps": steps,
            "notes": notes + [message],
        },
    )


def try_google_vision_dump(image_bytes: bytes) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Best-effort Google Vision dump for debugging. Returns the raw response as a JSON-serializable dict.

    This is optional: if credentials/library are not available we return (None, <reason>).
    """
    if GOOGLE_VISION_DISABLE:
        return None, "disabled"
    if _GOOGLE_ADC_ENV_ERROR:
        return None, _GOOGLE_ADC_ENV_ERROR
    try:
        from google.cloud import vision  # type: ignore
        from google.protobuf.json_format import MessageToDict  # type: ignore
    except Exception as exc:
        return None, f"google_cloud_vision_import_error:{exc}"
    try:
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        resp = client.document_text_detection(image=image)
        # If Vision returns an error payload, surface it.
        if getattr(resp, "error", None) and getattr(resp.error, "message", ""):
            return None, f"google_vision_error:{resp.error.message}"
        dump = MessageToDict(resp._pb)  # raw proto -> dict
        return dump, None
    except Exception as exc:
        return None, f"google_vision_call_error:{exc}"


def try_google_vision_dump_pages_from_pdf(
    pdf_bytes: bytes, max_pages: int, dpi: int
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Render PDF pages to images and run Google Vision Document OCR per page.

    Returns: (pages, error)
      pages[] = {"pageNumber", "width", "height", "dump"}
    """
    if GOOGLE_VISION_DISABLE:
        return [], "disabled"
    if _GOOGLE_ADC_ENV_ERROR:
        return [], _GOOGLE_ADC_ENV_ERROR
    try:
        from google.cloud import vision  # type: ignore
        from google.protobuf.json_format import MessageToDict  # type: ignore
    except Exception as exc:
        return [], f"google_cloud_vision_import_error:{exc}"
    try:
        from pdf2image import convert_from_bytes
    except Exception as exc:
        return [], f"pdf2image_import_error:{exc}"
    try:
        # Important: don't render the whole PDF if we're only OCR'ing the first N pages.
        # pdf2image is 1-indexed for first_page/last_page.
        if int(max_pages) <= 0:
            images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt="png")
        else:
            images = convert_from_bytes(
                pdf_bytes, dpi=dpi, fmt="png", first_page=1, last_page=max(1, int(max_pages))
            )
        if not images:
            return [], "pdf_render_failed_no_pages"
        client = vision.ImageAnnotatorClient()
        out: List[Dict[str, Any]] = []
        # If max_pages<=0 we treat it as "all pages rendered".
        page_limit = len(images) if int(max_pages) <= 0 else max(1, int(max_pages))
        for idx, img in enumerate(images[:page_limit], start=1):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            resp = client.document_text_detection(image=vision.Image(content=buf.getvalue()))
            if getattr(resp, "error", None) and getattr(resp.error, "message", ""):
                return out, f"google_vision_error:{resp.error.message}"
            out.append(
                {
                    "pageNumber": idx,
                    "width": int(getattr(img, "width", 0) or 0),
                    "height": int(getattr(img, "height", 0) or 0),
                    "dump": MessageToDict(resp._pb),
                }
            )
        return out, None
    except Exception as exc:
        return [], f"google_vision_call_error:{exc}"


def _vision_dump_page_to_words(vision_dump: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert Vision fullTextAnnotation->words into the same word shape used by our OCR pipeline.
    """
    words: List[Dict[str, Any]] = []
    fta = (vision_dump or {}).get("fullTextAnnotation") or {}
    pages = fta.get("pages") or []
    for p in pages:
        for block in p.get("blocks") or []:
            for para in block.get("paragraphs") or []:
                for w in para.get("words") or []:
                    syms = w.get("symbols") or []
                    text = "".join((s.get("text") or "") for s in syms).strip()
                    if not text:
                        continue
                    verts = (w.get("boundingBox") or {}).get("vertices") or []
                    xs = [int(v.get("x", 0) or 0) for v in verts] or [0]
                    ys = [int(v.get("y", 0) or 0) for v in verts] or [0]
                    x0, x1 = min(xs), max(xs)
                    y0, y1 = min(ys), max(ys)
                    words.append(
                        {
                            "text": text,
                            "x0": float(x0),
                            "x1": float(x1),
                            "top": float(y0),
                            "bottom": float(y1),
                        }
                    )
    return words


def _vision_dump_to_page_text(vision_dump: Dict[str, Any]) -> str:
    try:
        fta = (vision_dump or {}).get("fullTextAnnotation") or {}
        txt = fta.get("text") or ""
        return str(txt)
    except Exception:
        return ""


def _parse_pdf_with_google_vision(
    pdf_bytes: bytes,
    *,
    max_pages: int,
    dpi: int,
    steps: List[Dict[str, str]],
    include_dump: bool,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """
    Vision-first parsing path:
    - Render PDF pages -> PNG (pdf2image)
    - Run Vision Document OCR per page
    - Convert Vision words -> our page_data and reuse our row reconstruction
    """
    pages, err = try_google_vision_dump_pages_from_pdf(pdf_bytes, max_pages=max_pages, dpi=dpi)
    if err:
        add_step(steps, "google_vision", "failed", err)
        return [], None, err
    add_step(steps, "google_vision", "success", f"pages={len(pages)} dpi={dpi}")

    raw_rows: List[Dict[str, Any]] = []
    for p in pages:
        page_number = int(p.get("pageNumber") or 0) or 1
        dump = p.get("dump") or {}
        page_data = {
            "pageNumber": page_number,
            "width": float(p.get("width") or 0),
            "height": float(p.get("height") or 0),
            "words": _vision_dump_page_to_words(dump),
            "tables": [],
            "text": _vision_dump_to_page_text(dump),
        }

        # Vision output is OCR-like; the statement-column parser is more robust than generic geometry alone.
        page_rows = reconstruct_rows_statement_columns(page_data, source="google_vision")
        if not page_rows:
            page_rows = reconstruct_rows_geometric(page_data, source="google_vision")
        raw_rows.extend(page_rows)
        add_step(steps, "google_vision_page_parse", "success", f"page={page_number} rows={len(page_rows)}")

    dump_out: Optional[Dict[str, Any]] = None
    if include_dump:
        dump_out = {"dpi": dpi, "pages": pages}
    return raw_rows, dump_out, None


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


def _is_amount_token(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    if not re.search(r"\d", s):
        return False
    # Amount tokens should include decimals in bank tables; avoid capturing years / account numbers.
    return bool(re.fullmatch(r"(?:£|\$|€)?\s*[-(]?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}\)?", s))


def _infer_amount_column_centers(words: List[Dict[str, Any]], page_width: float) -> Tuple[float, float, float]:
    xs: List[float] = []
    for w in words:
        if _is_amount_token(str(w.get("text", ""))):
            x0 = float(w.get("x0", 0))
            x1 = float(w.get("x1", 0))
            xs.append((x0 + x1) / 2.0)
    if len(xs) < 8:
        return (page_width * 0.68, page_width * 0.82, page_width * 0.93)
    # Histogram peaks (simple, robust).
    bins = 30
    mn, mx = min(xs), max(xs)
    if mx - mn < 1:
        return (page_width * 0.68, page_width * 0.82, page_width * 0.93)
    counts = [0] * bins
    for x in xs:
        idx = int((x - mn) / (mx - mn) * (bins - 1))
        counts[idx] += 1
    # Pick top 3 bins with separation.
    picked = []
    for idx in sorted(range(bins), key=lambda i: counts[i], reverse=True):
        if counts[idx] == 0:
            break
        if all(abs(idx - p) >= 3 for p in picked):
            picked.append(idx)
        if len(picked) == 3:
            break
    if len(picked) < 3:
        return (page_width * 0.68, page_width * 0.82, page_width * 0.93)
    centers = sorted(mn + (p + 0.5) * (mx - mn) / bins for p in picked)
    # left=paid_out, middle=paid_in, right=balance
    return (centers[0], centers[1], centers[2])


def reconstruct_rows_statement_columns(page_data: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
    """
    OCR/table-style parsing:
    - date shown once, subsequent rows inherit date
    - payment type (DD/VIS/BP/...) column
    - paid out / paid in / balance numeric columns on the right
    """
    rows: List[Dict[str, Any]] = []
    page_number = int(page_data.get("pageNumber", 1))
    width = float(page_data.get("width", 0))
    words = page_data.get("words", []) or []

    out_x, in_x, bal_x = _infer_amount_column_centers(words, width)

    # Anchor-based row splitting: each payment-type token becomes a row anchor.
    anchors: List[Dict[str, Any]] = []
    for w in words:
        txt = str(w.get("text", "")).strip()
        if not txt:
            continue
        if PAYMENT_TYPE_REGEX.fullmatch(txt):
            anchors.append({"top": float(w.get("top", 0)), "kind": "ptype", "value": txt.upper()})
            continue
        # Contactless markers are often OCR'd as ')', '))', or ')))'
        if txt in (")", "))", ")))"):
            x0 = float(w.get("x0", 0))
            if x0 < width * 0.45:
                anchors.append({"top": float(w.get("top", 0)), "kind": "ptype", "value": "CONTACTLESS"})
            continue
        # Amount anchors (paid-out/paid-in columns) are more reliable than payment type tokens.
        if _is_amount_token(txt):
            cx = (float(w.get("x0", 0)) + float(w.get("x1", 0))) / 2.0
            nearest = min(
                [("out", out_x), ("in", in_x), ("bal", bal_x)],
                key=lambda kv: abs(cx - kv[1]),
            )[0]
            if nearest in ("out", "in"):
                anchors.append({"top": float(w.get("top", 0)), "kind": "amount", "value": nearest})
            elif nearest == "bal":
                anchors.append({"top": float(w.get("top", 0)), "kind": "balance_amt", "value": "bal"})

    # Add explicit balance anchors if present.
    full_text = (page_data.get("text") or "").lower()
    if "balance brought forward" in full_text or "balance carried forward" in full_text:
        # We'll still detect via band text, but ensure there is an anchor near the first occurrence.
        for w in words:
            if str(w.get("text", "")).lower() == "balance":
                anchors.append({"top": float(w.get("top", 0)), "kind": "balance", "value": "BALANCE"})
                break

    # De-dupe anchors by y proximity.
    anchors = sorted(anchors, key=lambda a: a["top"])
    dedup = []
    for a in anchors:
        if not dedup or abs(a["top"] - dedup[-1]["top"]) > 3:
            dedup.append(a)
    anchors = dedup
    if len(anchors) < 3:
        # Fallback to generic line-based parse for sparse OCR.
        lines = _group_words_into_lines(words, y_tol=9.0)
        for line in lines:
            text = _line_text(line)
            if text:
                pass
        return rows

    current_date = ""
    for i, a in enumerate(anchors):
        top = float(a["top"])
        next_top = float(anchors[i + 1]["top"]) if i + 1 < len(anchors) else 1e9
        band_words = [w for w in words if top - 10 <= float(w.get("top", 0)) < next_top - 3]
        if not band_words:
            continue
        band_sorted = sorted(band_words, key=lambda w: float(w.get("x0", 0)))
        band_text = _words_to_text_yx(band_words)
        if not band_text:
            continue
        # Balance lines include the word "Balance" and would look like headers otherwise.
        if "balance brought forward" not in band_text.lower() and "balance carried forward" not in band_text.lower():
            if _is_probable_header(band_text):
                continue

        date_match = DATE_REGEX.search(band_text)
        if date_match:
            current_date = date_match.group(1)

        # Determine payment type from band content (more robust than anchor value).
        payment_type = "UNK"
        for w in band_words:
            t = str(w.get("text", "")).strip()
            if PAYMENT_TYPE_REGEX.fullmatch(t):
                payment_type = t.upper()
                break
        if payment_type == "UNK":
            if any(str(w.get("text", "")).strip() in (")", "))", ")))") and float(w.get("x0", 0)) < width * 0.45 for w in band_words):
                payment_type = "CONTACTLESS"

        # Amounts by column.
        paid_out_raw = ""
        paid_in_raw = ""
        balance_raw = ""
        for w in band_sorted:
            wt = str(w.get("text", "")).strip()
            if not _is_amount_token(wt):
                continue
            cx = (float(w.get("x0", 0)) + float(w.get("x1", 0))) / 2.0
            nearest = min(
                [("out", out_x), ("in", in_x), ("bal", bal_x)],
                key=lambda kv: abs(cx - kv[1]),
            )[0]
            if nearest == "out":
                paid_out_raw = wt
            elif nearest == "in":
                paid_in_raw = wt
            else:
                balance_raw = wt

        if "balance brought forward" in band_text.lower():
            rows.append(
                {
                    "source": source,
                    "pageNumber": page_number,
                    "dateRaw": current_date,
                    "paymentType": "BALANCE",
                    "descriptionRaw": "Balance brought forward",
                    "paidOutRaw": "",
                    "paidInRaw": "",
                    "balanceRaw": balance_raw,
                    "originalSourceText": band_text,
                    "_yTop": top,
                }
            )
            continue
        if "balance carried forward" in band_text.lower():
            rows.append(
                {
                    "source": source,
                    "pageNumber": page_number,
                    "dateRaw": current_date,
                    "paymentType": "BALANCE",
                    "descriptionRaw": "Balance carried forward",
                    "paidOutRaw": "",
                    "paidInRaw": "",
                    "balanceRaw": balance_raw,
                    "originalSourceText": band_text,
                    "_yTop": top,
                }
            )
            continue

        # Details: strip date + payment type + numeric tokens.
        details = band_text
        if current_date:
            details = re.sub(re.escape(current_date), " ", details, count=1)
        if payment_type and payment_type != "CONTACTLESS":
            details = re.sub(rf"\\b{re.escape(payment_type)}\\b", " ", details, count=1, flags=re.I)
        if payment_type == "CONTACTLESS":
            details = details.replace(")))", " ")
        details = re.sub(r"\\b\\d{1,3}(?:,\\d{3})*\\.\\d{2}\\b", " ", details)
        details = _strip_noise_tokens(details)

        rows.append(
            {
                "source": source,
                "pageNumber": page_number,
                "dateRaw": current_date,
                "paymentType": payment_type,
                "descriptionRaw": details,
                "paidOutRaw": paid_out_raw,
                "paidInRaw": paid_in_raw,
                "balanceRaw": balance_raw,
                "originalSourceText": band_text,
                "_yTop": top,
            }
        )

    rows = _merge_statement_rows(rows)
    for r in rows:
        r.pop("_yTop", None)
    return rows


def _line_text(line_words: List[Dict[str, Any]]) -> str:
    return " ".join(
        str(w.get("text", "")).strip() for w in sorted(line_words, key=lambda w: float(w.get("x0", 0)))
        if str(w.get("text", "")).strip()
    )


def _words_to_text_yx(words: List[Dict[str, Any]]) -> str:
    return " ".join(
        str(w.get("text", "")).strip()
        for w in sorted(words, key=lambda w: (float(w.get("top", 0)), float(w.get("x0", 0))))
        if str(w.get("text", "")).strip()
    )


def _merge_statement_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge OCR-split table rows.

    Common OCR pattern on scanned statements:
    - Row A: paymentType + merchant (no amount)
    - Row B: location/continuation + amount (UNK paymentType)
    These are the same transaction; merge them.
    """
    merged: List[Dict[str, Any]] = []
    i = 0
    while i < len(rows):
        r = rows[i]
        if i + 1 < len(rows):
            n = rows[i + 1]
            r_pt = str(r.get("paymentType") or "").strip().upper()
            n_pt = str(n.get("paymentType") or "").strip().upper()
            r_has_amt = bool(r.get("paidOutRaw") or r.get("paidInRaw"))
            n_has_amt = bool(n.get("paidOutRaw") or n.get("paidInRaw"))
            same_date = (r.get("dateRaw") or "") == (n.get("dateRaw") or "")
            close_y = abs(float(r.get("_yTop", 0)) - float(n.get("_yTop", 0))) <= 45

            if (
                same_date
                and close_y
                and r_pt not in ("", "UNK", "BALANCE")
                and not r_has_amt
                and n_pt in ("", "UNK")
                and n_has_amt
            ):
                r["descriptionRaw"] = " ".join(f"{r.get('descriptionRaw','')} {n.get('descriptionRaw','')}".split())
                r["originalSourceText"] = " ".join(
                    f"{r.get('originalSourceText','')} {n.get('originalSourceText','')}".split()
                ).strip()
                for k in ("paidOutRaw", "paidInRaw", "balanceRaw"):
                    if not r.get(k) and n.get(k):
                        r[k] = n.get(k)
                merged.append(r)
                i += 2
                continue

        merged.append(r)
        i += 1

    # Propagate payment type when OCR misses it (common on scanned tables).
    for j in range(1, len(merged)):
        cur = merged[j]
        prev = merged[j - 1]
        cur_pt = str(cur.get("paymentType") or "").strip().upper()
        prev_pt = str(prev.get("paymentType") or "").strip().upper()
        if cur_pt not in ("", "UNK") or prev_pt in ("", "UNK", "BALANCE"):
            continue
        if (cur.get("dateRaw") or "") != (prev.get("dateRaw") or ""):
            continue
        if abs(float(cur.get("_yTop", 0)) - float(prev.get("_yTop", 0))) > 70:
            continue
        cur["paymentType"] = prev_pt

    return merged


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
        # Prefer statement-column reconstruction for OCR output; it handles repeated-date blocks.
        parsed = reconstruct_rows_statement_columns(page_data, source="ocr")
        if parsed:
            return parsed, f"ocr_applied:page={page_number},words={extracted_words},mode=columns,reasons={','.join(reasons)}"
        return reconstruct_rows_geometric(page_data, source="ocr"), (
            f"ocr_applied:page={page_number},words={extracted_words},mode=generic,reasons={','.join(reasons)}"
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
        parsed = reconstruct_rows_statement_columns(page_data, source="ocr_image")
        if parsed:
            return parsed, f"ocr_image_words={extracted_words},mode=columns"
        return reconstruct_rows_geometric(page_data, source="ocr_image"), f"ocr_image_words={extracted_words},mode=generic"
    except Exception as exc:
        return [], f"ocr_image_failed:{exc}"


def _parse_json_object_loose(raw: str) -> Dict[str, Any]:
    s = (raw or "").strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try fenced code extraction.
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            return {}
    # Try first/last brace window.
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b != -1 and b > a:
        try:
            return json.loads(s[a : b + 1])
        except Exception:
            return {}
    return {}


def _compute_confidence(
    row: Dict[str, Any],
    date_value: Optional[str],
    amount_value: Optional[float],
    description: str,
    reference: str,
) -> float:
    score = 0.0
    if date_value:
        score += 0.35
    if amount_value is not None:
        score += 0.35
    desc_len = len(description)
    if desc_len >= 16:
        score += 0.12
    elif desc_len >= 8:
        score += 0.06
    if reference:
        score += 0.05
    if DATE_REGEX.search(str(row.get("descriptionRaw", ""))):
        score += 0.05
    if AMOUNT_REGEX.search(str(row.get("descriptionRaw", ""))):
        score += 0.05
    return round(min(1.0, max(0.0, score)), 2)


def _strip_noise_tokens(text: str) -> str:
    t = text
    # Drop common statement column labels and trailing fragments.
    t = re.sub(r"\bDate\b", " ", t, flags=re.I)
    t = re.sub(r"\bDescription\b", " ", t, flags=re.I)
    t = re.sub(r"\bType\b", " ", t, flags=re.I)
    t = re.sub(r"\bMoney\s*Out(?:\s*\(.*?\))?\b", " ", t, flags=re.I)
    t = re.sub(r"\bMoney\s*In(?:\s*\(.*?\))?\b", " ", t, flags=re.I)
    t = re.sub(r"\bDebit\b|\bCredit\b|\bAmount\b", " ", t, flags=re.I)
    t = re.sub(r"\bBalance(?:\s*on)?(?:\s*\(.*?\))?\b.*$", " ", t, flags=re.I)
    t = re.sub(r"\bblank\b", " ", t, flags=re.I)
    t = re.sub(r"[.]{2,}", " ", t)
    return " ".join(t.split()).strip()


def _derive_transaction_description(description_raw: str, date_raw: str, amount_raw: str) -> str:
    original = " ".join((description_raw or "").split())
    if not original:
        return ""

    # Best structured pattern first: Description ... Type ...
    m = re.search(
        r"\bDescription\b\s*(.*?)\s*(?:\bType\b|\bMoney\s*In\b|\bMoney\s*Out\b|\bBalance\b)",
        original,
        flags=re.I,
    )
    if m and m.group(1).strip():
        return " ".join(m.group(1).split())

    text = original

    # Remove leading explicit date token.
    if date_raw:
        dr = re.escape(" ".join(date_raw.split()))
        text = re.sub(rf"^\s*{dr}\s*", " ", text, flags=re.I)
        text = re.sub(rf"^\s*Date\s*{dr}\s*", " ", text, flags=re.I)
    else:
        text = re.sub(rf"^\s*{DATE_REGEX.pattern}\s*", " ", text, flags=re.I)

    # Remove transaction amount token first (keeps payee phrase cleaner).
    if amount_raw:
        ar = re.escape(" ".join(str(amount_raw).split()))
        text = re.sub(rf"\s*{ar}\s*", " ", text, flags=re.I)

    # Remove other trailing amount/balance artifacts.
    text = re.sub(r"\b(?:£|\$|€)\s*[-(]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})\)?(?:\s*(?:CR|DR))?\b", " ", text)
    text = _strip_noise_tokens(text)
    # Running balances often remain as a trailing number.
    text = re.sub(r"\s+(?:£|\$|€)?[-(]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})\)?\s*$", " ", text)
    text = " ".join(text.split()).strip()
    return text if text else original


def repair_rows_with_gpt_compact(
    headers: List[str],
    sample_rows: List[Dict[str, Any]],
    failed_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    if not OPENAI_API_KEY:
        return [], "gpt_skipped_no_api_key", []
    try:
        from openai import OpenAI
    except Exception as exc:
        return [], f"gpt_skipped_openai_import_error:{exc}", []

    compact_sample = sample_rows[:GPT_MAX_SAMPLE_ROWS]
    compact_failed = failed_rows[:GPT_MAX_SAMPLE_ROWS]
    payload = {
        "headers": headers,
        "sample_rows": compact_sample,
        "failed_rows": compact_failed,
    }
    prompt = (
        "You are repairing bank statement row extraction.\n"
        "Return STRICT JSON object with keys:\n"
        "repaired_rows: array of rows with fields {pageNumber,dateRaw,amountRaw,descriptionRaw,referenceRaw,source}\n"
        "notes: array of short strings\n"
        "sanity_warnings: array of short strings\n"
        "Only include rows you are confident about. Do not invent rows.\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=True)}"
    )
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=GPT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": "You output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        obj = _parse_json_object_loose(content)
        rows = obj.get("repaired_rows", []) if isinstance(obj, dict) else []
        notes = []
        if isinstance(obj, dict):
            for k in ("notes", "sanity_warnings"):
                v = obj.get(k, [])
                if isinstance(v, list):
                    notes.extend(str(x) for x in v[:20])
        repaired: List[Dict[str, Any]] = []
        for r in rows if isinstance(rows, list) else []:
            if not isinstance(r, dict):
                continue
            if not (r.get("dateRaw") and r.get("amountRaw") and r.get("descriptionRaw")):
                continue
            repaired.append(
                {
                    "source": str(r.get("source") or "gpt_repair"),
                    "pageNumber": int(r.get("pageNumber") or 0),
                    "dateRaw": str(r.get("dateRaw")),
                    "amountRaw": str(r.get("amountRaw")),
                    "descriptionRaw": str(r.get("descriptionRaw")),
                    "referenceRaw": str(r.get("referenceRaw") or ""),
                }
            )
        return repaired, f"gpt_repair_used model={GPT_MODEL} repaired={len(repaired)}", notes
    except Exception as exc:
        return [], f"gpt_repair_failed:{exc}", []


def validate_transactions_with_gpt_compact(transactions: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    if not OPENAI_API_KEY:
        return "gpt_validation_skipped_no_api_key", []
    try:
        from openai import OpenAI
    except Exception as exc:
        return f"gpt_validation_skipped_openai_import_error:{exc}", []

    compact = transactions[:GPT_MAX_SAMPLE_ROWS]
    prompt = (
        "Review these parsed transactions for likely extraction issues.\n"
        "Return STRICT JSON object with key validation_notes as array of short notes.\n"
        f"transactions={json.dumps(compact, ensure_ascii=True)}"
    )
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=GPT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": "You output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        obj = _parse_json_object_loose(resp.choices[0].message.content or "{}")
        notes = obj.get("validation_notes", []) if isinstance(obj, dict) else []
        if not isinstance(notes, list):
            notes = []
        return f"gpt_validation_used model={GPT_MODEL}", [str(n) for n in notes[:20]]
    except Exception as exc:
        return f"gpt_validation_failed:{exc}", []


def normalize_transactions(raw_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    transactions: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for row in raw_rows:
        date_value = parse_date(str(row.get("dateRaw", "")).strip())

        payment_type = str(row.get("paymentType") or "").strip() or None
        paid_out = parse_amount(str(row.get("paidOutRaw", "")).strip()) if row.get("paidOutRaw") else None
        paid_in = parse_amount(str(row.get("paidInRaw", "")).strip()) if row.get("paidInRaw") else None
        balance = parse_amount(str(row.get("balanceRaw", "")).strip()) if row.get("balanceRaw") else None

        # Balance forward/carry rows are useful even without a transaction amount.
        if (payment_type or "").upper() == "BALANCE" and date_value and balance is not None:
            desc = str(row.get("descriptionRaw") or "Balance").strip()
            tx = {
                "id": str(uuid.uuid4()),
                "type": "income",
                "date": date_value,
                "payeePayer": "Balance",
                "description": desc,
                "transactionDescription": desc,
                "originalSourceText": " ".join(str(row.get("originalSourceText") or "").split()),
                "paymentType": "BALANCE",
                "paidOutGbp": None,
                "paidInGbp": None,
                "balanceGbp": balance,
                "reference": "",
                "amount": 0.0,
                "category": "Bank statement import",
                "confidence": 0.95,
            }
            transactions.append(tx)
            continue

        # Primary amount used for income/expense classification.
        amount_value = None
        if paid_out is not None and paid_out != 0:
            amount_value = -abs(paid_out)
        elif paid_in is not None and paid_in != 0:
            amount_value = abs(paid_in)
        else:
            amount_value = parse_amount(str(row.get("amountRaw", "")).strip())

        original_source_text = " ".join(str(row.get("originalSourceText") or row.get("descriptionRaw", "")).split())
        description = _derive_transaction_description(
            original_source_text,
            str(row.get("dateRaw", "")).strip(),
            str(row.get("amountRaw", "")).strip() or str(row.get("paidOutRaw", "")).strip() or str(row.get("paidInRaw", "")).strip(),
        )
        if _is_summary_non_transaction_line(description):
            failed.append({**row, "_reason": "summary_line"})
            continue
        if not date_value or amount_value is None or not description:
            failed.append({**row, "_reason": "missing_core_fields"})
            continue
        reference = ""
        ref_match = re.search(r"\b(ref(?:erence)?[:\s-]*[A-Z0-9-]{4,})\b", description, re.I)
        if ref_match:
            reference = ref_match.group(1).strip()
        d = f"{description} {original_source_text}".lower()
        if amount_value < 0:
            tx_type = "expense"
        elif re.search(r"\bdeb(it)?\b|\bdr\b|money out|withdrawal|card|purchase|payment to|transfer out", d):
            tx_type = "expense"
        elif re.search(r"\bcr\b|credit|money in|salary|refund|interest|reward|transfer in", d):
            tx_type = "income"
        else:
            tx_type = "income"
        confidence = _compute_confidence(row, date_value, amount_value, description, reference)
        tx = {
            "id": str(uuid.uuid4()),
            "type": tx_type,
            "date": date_value,
            "payeePayer": description[:120],
            "description": description,
            "transactionDescription": description,
            "originalSourceText": original_source_text,
            "paymentType": payment_type,
            "paidOutGbp": abs(paid_out) if paid_out is not None else None,
            "paidInGbp": abs(paid_in) if paid_in is not None else None,
            "balanceGbp": balance,
            "reference": reference,
            "amount": abs(round(float(amount_value), 2)),
            "category": "Bank statement import",
            "confidence": confidence,
        }
        transactions.append(tx)
    return transactions, failed


@app.get("/parse-bank-statement/viewer", response_class=HTMLResponse)
def parse_bank_statement_viewer() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>TaxPig Parser Viewer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    table { border-collapse: collapse; width: 100%; margin-top: 12px; }
    th, td { border: 1px solid #ddd; padding: 6px; font-size: 12px; }
    th { background: #f4f4f4; text-align: left; }
    .muted { color: #666; font-size: 12px; }
    pre { background: #111; color: #eee; padding: 10px; overflow: auto; }
  </style>
</head>
<body>
  <h2>Bank Statement Parser Viewer</h2>
  <p class="muted">Upload PDF/PNG/JPG and render parsed transactions table from <code>/parse-bank-statement</code>.</p>
  <form id="f">
    <label>Bearer token: <input id="token" style="width:420px" /></label><br/><br/>
    <input type="file" id="file" required />
    <button type="submit">Parse</button>
  </form>
  <div id="summary" class="muted"></div>
  <table id="tbl" hidden>
    <thead><tr>
      <th>Date</th><th>Payment Type</th><th>Paid out</th><th>Paid in</th><th>Balance</th><th>Type</th><th>Amount</th><th>Payee/Payer</th><th>Transaction Description</th><th>Original Source Text</th><th>Reference</th><th>Confidence</th>
    </tr></thead>
    <tbody></tbody>
  </table>
  <h4>Steps/Notes</h4>
  <pre id="meta"></pre>
  <script>
    const form = document.getElementById('f');
    const tbl = document.getElementById('tbl');
    const tbody = tbl.querySelector('tbody');
    const summary = document.getElementById('summary');
    const meta = document.getElementById('meta');
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      tbody.innerHTML = '';
      tbl.hidden = true;
      summary.textContent = 'Parsing...';
      const fd = new FormData();
      const file = document.getElementById('file').files[0];
      fd.append('file', file);
      const token = document.getElementById('token').value.trim();
      const headers = {};
      if (token) headers['Authorization'] = 'Bearer ' + token;
      const res = await fetch('/parse-bank-statement', { method: 'POST', body: fd, headers });
      const data = await res.json();
      const tx = Array.isArray(data.transactions) ? data.transactions : [];
      summary.textContent = `ok=${data.ok} transactions=${tx.length}`;
      for (const t of tx) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${t.date||''}</td><td>${t.paymentType||''}</td><td>${t.paidOutGbp??''}</td><td>${t.paidInGbp??''}</td><td>${t.balanceGbp??''}</td><td>${t.type||''}</td><td>${t.amount??''}</td><td>${(t.payeePayer||'').replace(/</g,'&lt;')}</td><td>${(t.transactionDescription||t.description||'').replace(/</g,'&lt;')}</td><td>${(t.originalSourceText||'').replace(/</g,'&lt;')}</td><td>${(t.reference||'').replace(/</g,'&lt;')}</td><td>${t.confidence??''}</td>`;
        tbody.appendChild(tr);
      }
      tbl.hidden = tx.length === 0;
      meta.textContent = JSON.stringify({steps:data.steps, notes:data.notes}, null, 2);
    });
  </script>
</body>
</html>"""


def _json_from_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, JSONResponse):
        try:
            return json.loads(result.body.decode("utf-8"))
        except Exception:
            return {"ok": False, "transactions": [], "steps": [], "notes": ["Invalid JSONResponse body"]}
    if isinstance(result, dict):
        return result
    return {"ok": False, "transactions": [], "steps": [], "notes": [f"Unexpected result type: {type(result)}"]}


def _transactions_to_statement_csv(transactions: List[Dict[str, Any]]) -> str:
    import io
    import csv

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["date", "payment_type", "description", "paid_out_gbp", "paid_in_gbp", "balance_gbp"])
    for t in transactions:
        w.writerow(
            [
                t.get("date") or "",
                t.get("paymentType") or "",
                t.get("transactionDescription") or t.get("description") or "",
                "" if t.get("paidOutGbp") is None else f"{t.get('paidOutGbp'):.2f}",
                "" if t.get("paidInGbp") is None else f"{t.get('paidInGbp'):.2f}",
                "" if t.get("balanceGbp") is None else f"{t.get('balanceGbp'):.2f}",
            ]
        )
    return buf.getvalue()


@app.post("/parse-bank-statement/from-url")
async def parse_bank_statement_from_url(
    url: str,
    authorization: Optional[str] = Header(default=None),
    includeVisionDump: bool = False,
    useVision: bool = False,
    visionMaxPages: Optional[int] = None,
    visionDpi: int = GOOGLE_VISION_DEFAULT_DPI,
):
    steps: List[Dict[str, str]] = []
    notes: List[str] = []
    try:
        if not url.lower().startswith(("http://", "https://")):
            add_step(steps, "download", "failed", "URL must start with http:// or https://")
            return build_error_response("Invalid URL.", steps, notes, status_code=400)

        # Some Railway base images lack a system CA bundle; force Requests to use certifi.
        try:
            import certifi  # type: ignore

            ca_path = certifi.where()
        except Exception:
            ca_path = True  # fall back to requests default

        resp = requests.get(url, timeout=90, verify=ca_path)
        if resp.status_code >= 400:
            add_step(steps, "download", "failed", f"HTTP {resp.status_code}")
            return build_error_response(f"Failed to download URL (HTTP {resp.status_code}).", steps, notes, 400)
        content = resp.content or b""
        if not content:
            add_step(steps, "download", "failed", "Downloaded body is empty")
            return build_error_response("Downloaded file is empty.", steps, notes, 400)
        if len(content) > MAX_UPLOAD_BYTES:
            add_step(steps, "download", "failed", f"Downloaded file exceeds {MAX_UPLOAD_MB}MB")
            return build_error_response(f"File too large (max {MAX_UPLOAD_MB}MB).", steps, notes, 413)

        parsed = urlparse(url)
        guessed_name = os.path.basename(unquote(parsed.path)) or "downloaded.pdf"
        guessed_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        add_step(steps, "download", "success", f"Fetched {guessed_name} ({len(content)} bytes)")

        tmp = SpooledTemporaryFile(max_size=MAX_UPLOAD_BYTES)
        tmp.write(content)
        tmp.seek(0)
        headers = Headers({"content-type": guessed_type or "application/octet-stream"})
        uf = StarletteUploadFile(file=tmp, filename=guessed_name, headers=headers)
        result = await parse_bank_statement(
            file=uf,
            authorization=authorization,
            includeVisionDump=includeVisionDump,
            useVision=useVision,
            visionMaxPages=GOOGLE_VISION_DEFAULT_MAX_PAGES if visionMaxPages is None else int(visionMaxPages),
            visionDpi=visionDpi,
        )
        payload = _json_from_result(result)
        if isinstance(payload, dict) and "transactions" in payload:
            payload["statementCsv"] = _transactions_to_statement_csv(list(payload.get("transactions") or []))
        payload_steps = payload.get("steps", [])
        payload_steps.insert(0, {"service": "download", "status": "success", "detail": f"url={url}"})
        payload["steps"] = payload_steps
        return payload
    except Exception as exc:
        add_step(steps, "download", "failed", str(exc))
        return build_error_response("URL parse failed.", steps, notes + [str(exc)], 500)


@app.get("/parse-bank-statement/quick-view", response_class=HTMLResponse)
async def parse_bank_statement_quick_view(
    url: str,
    token: Optional[str] = None,
    visionMaxPages: Optional[int] = None,
    visionDpi: int = GOOGLE_VISION_DEFAULT_DPI,
    only: Optional[str] = None,
) -> str:
    auth = f"Bearer {token}" if token else None
    effective_vision_max_pages = GOOGLE_VISION_DEFAULT_MAX_PAGES if visionMaxPages is None else int(visionMaxPages)

    def _render_table(transactions: List[Dict[str, Any]]) -> str:
        rows = []
        for t in (transactions or [])[:1000]:
            rows.append(
                "<tr>"
                f"<td>{t.get('date','')}</td>"
                f"<td>{t.get('paymentType','')}</td>"
                f"<td>{t.get('paidOutGbp','')}</td>"
                f"<td>{t.get('paidInGbp','')}</td>"
                f"<td>{t.get('balanceGbp','')}</td>"
                f"<td>{t.get('type','')}</td>"
                f"<td>{t.get('amount','')}</td>"
                f"<td>{str(t.get('payeePayer','')).replace('<','&lt;')}</td>"
                f"<td>{str(t.get('transactionDescription') or t.get('description','')).replace('<','&lt;')}</td>"
                f"<td>{str(t.get('originalSourceText','')).replace('<','&lt;')}</td>"
                f"<td>{str(t.get('reference','')).replace('<','&lt;')}</td>"
                f"<td>{t.get('confidence','')}</td>"
                "</tr>"
            )
        return (
            "<table><thead><tr>"
            "<th>Date</th><th>Payment Type</th><th>Paid out</th><th>Paid in</th><th>Balance</th><th>Type</th>"
            "<th>Amount</th><th>Payee/Payer</th><th>Transaction Description</th><th>Original Source Text</th>"
            "<th>Reference</th><th>Confidence</th>"
            "</tr></thead><tbody>"
            + ("".join(rows) if rows else '<tr><td colspan="12">No transactions</td></tr>')
            + "</tbody></table>"
        )

    only_norm = (only or "").strip().lower()
    show_pdfplumber = only_norm in {"", "both", "all", "pdf", "pdfplumber"}
    show_vision = only_norm in {"", "both", "all", "vision", "google", "googlevision"}
    if only_norm == "vision":
        show_pdfplumber = False
        show_vision = True
    if only_norm in {"pdf", "pdfplumber"}:
        show_pdfplumber = True
        show_vision = False

    # Run only what the caller asked for (Vision can be expensive).
    pdfplumber_result = None
    vision_result = None
    if show_pdfplumber:
        pdfplumber_result = await parse_bank_statement_from_url(
            url=url,
            authorization=auth,
            includeVisionDump=False,
            useVision=False,
        )
    if show_vision:
        vision_result = await parse_bank_statement_from_url(
            url=url,
            authorization=auth,
            includeVisionDump=True,
            useVision=True,
            visionMaxPages=effective_vision_max_pages,
            visionDpi=visionDpi,
        )

    pdfplumber_payload = _json_from_result(pdfplumber_result) if pdfplumber_result is not None else {"ok": True, "transactions": [], "steps": [], "notes": []}
    vision_payload = _json_from_result(vision_result) if vision_result is not None else {"ok": True, "transactions": [], "steps": [], "notes": []}

    pdfplumber_tx = pdfplumber_payload.get("transactions", []) if isinstance(pdfplumber_payload, dict) else []
    vision_tx = vision_payload.get("transactions", []) if isinstance(vision_payload, dict) else []

    pdfplumber_meta = json.dumps(
        {"ok": pdfplumber_payload.get("ok"), "steps": pdfplumber_payload.get("steps", []), "notes": pdfplumber_payload.get("notes", [])},
        indent=2,
    )
    vision_meta = json.dumps(
        {"ok": vision_payload.get("ok"), "steps": vision_payload.get("steps", []), "notes": vision_payload.get("notes", [])},
        indent=2,
    )

    pdfplumber_csv = str(pdfplumber_payload.get("statementCsv") or "")
    vision_csv = str(vision_payload.get("statementCsv") or "")

    vision_err = vision_payload.get("googleVisionDumpError")
    vision_dump_preview = ""
    vision_dump = vision_payload.get("googleVisionDump")
    if vision_dump is not None:
        try:
            # Don't inline the full dump in HTML (it can be huge). Show a preview + a download link.
            blob = json.dumps(vision_dump, indent=2)
            vision_dump_preview = blob[:20000] + ("\n... (truncated preview; use download link for full JSON)" if len(blob) > 20000 else "")
        except Exception:
            vision_dump_preview = str(vision_dump)[:20000]
    elif vision_err:
        vision_dump_preview = f"(not available: {vision_err})"

    # Preserve the token for convenience in download links.
    token_q = f"&token={token}" if token else ""
    vision_dump_download_url = (
        f"/parse-bank-statement/vision-dump?url={requests.utils.quote(url, safe='')}{token_q}"
        f"&visionMaxPages={int(effective_vision_max_pages)}&visionDpi={int(visionDpi)}"
    )

    tabs_html = ""
    if show_pdfplumber and show_vision:
        tabs_html = """
<div class="tabs">
  <button class="tabbtn active" data-tab="pdfplumber">pdfplumber</button>
  <button class="tabbtn" data-tab="vision">google vision</button>
</div>
"""

    top_links = ""
    if show_vision:
        top_links = f'Vision dump download: <a href="{vision_dump_download_url}">/parse-bank-statement/vision-dump</a>'

    # Default active panel depends on "only".
    pdf_active = "active" if (show_pdfplumber and (not show_vision or (show_pdfplumber and show_vision))) else ""
    vision_active = "active" if (show_vision and not show_pdfplumber) else ""

    pdf_section_html = ""
    if show_pdfplumber:
        pdf_section_html = f"""
<div id="pdfplumber" class="panel {pdf_active}">
  <h4>pdfplumber-first parse</h4>
  <p class="muted">Rows shown: {len(pdfplumber_tx)}</p>
  {_render_table(pdfplumber_tx)}
  <h4>Statement CSV</h4><pre>{pdfplumber_csv.replace('<','&lt;')}</pre>
  <h4>Steps / Notes</h4><pre>{pdfplumber_meta}</pre>
</div>
"""

    vision_section_html = ""
    if show_vision:
        vision_section_html = f"""
<div id="vision" class="panel {vision_active}">
  <h4>Google Vision parse (visionMaxPages={int(effective_vision_max_pages)} dpi={int(visionDpi)})</h4>
  <p class="muted">Rows shown: {len(vision_tx)}</p>
  {_render_table(vision_tx)}
  <h4>Google Vision Dump (preview)</h4><pre>{vision_dump_preview.replace('<','&lt;')}</pre>
  <h4>Statement CSV</h4><pre>{vision_csv.replace('<','&lt;')}</pre>
  <h4>Steps / Notes</h4><pre>{vision_meta}</pre>
</div>
"""

    return f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>TaxPig Quick View</title>
<style>
  body{{font-family:Arial;margin:18px}}
  table{{border-collapse:collapse;width:100%}}
  th,td{{border:1px solid #ddd;padding:6px;font-size:12px;vertical-align:top}}
  th{{background:#f3f3f3}}
  pre{{background:#111;color:#eee;padding:8px;overflow:auto;max-height:520px}}
  .tabs{{display:flex;gap:8px;margin:12px 0}}
  .tabbtn{{padding:8px 10px;border:1px solid #bbb;border-radius:6px;background:#fafafa;cursor:pointer}}
  .tabbtn.active{{background:#111;color:#fff;border-color:#111}}
  .panel{{display:none}}
  .panel.active{{display:block}}
  .muted{{color:#666;font-size:12px}}
</style>
</head><body>
<h3>Quick View: {url}</h3>
<div class="muted">
  {top_links}
</div>
{tabs_html}

{pdf_section_html}
{vision_section_html}

<script>
  const btns = Array.from(document.querySelectorAll('.tabbtn'));
  if (btns.length) {{
    const panels = Array.from(document.querySelectorAll('.panel'));
    for (const b of btns) {{
      b.addEventListener('click', () => {{
        for (const x of btns) x.classList.toggle('active', x === b);
        const id = b.getAttribute('data-tab');
        for (const p of panels) p.classList.toggle('active', p.id === id);
      }});
    }}
  }}
</script>
</body></html>"""


@app.get("/parse-bank-statement/vision-dump")
async def parse_bank_statement_vision_dump(
    url: str,
    token: Optional[str] = None,
    authorization: Optional[str] = Header(default=None),
    visionMaxPages: Optional[int] = None,
    visionDpi: int = GOOGLE_VISION_DEFAULT_DPI,
):
    """
    Convenience endpoint for debugging: returns the exact Vision API response (JSON) for a URL.
    """
    auth = authorization or (f"Bearer {token}" if token else None)
    result = await parse_bank_statement_from_url(
        url=url,
        authorization=auth,
        includeVisionDump=True,
        useVision=True,
        visionMaxPages=GOOGLE_VISION_DEFAULT_MAX_PAGES if visionMaxPages is None else int(visionMaxPages),
        visionDpi=visionDpi,
    )
    payload = _json_from_result(result)
    dump = payload.get("googleVisionDump")
    err = payload.get("googleVisionDumpError")
    if dump is None:
        return JSONResponse(status_code=400, content={"ok": False, "error": err or "no_dump"})
    return JSONResponse(content=dump)


@app.post("/parse-bank-statement")
async def parse_bank_statement(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
    includeVisionDump: bool = False,
    useVision: bool = False,
    visionMaxPages: Optional[int] = None,
    visionDpi: int = GOOGLE_VISION_DEFAULT_DPI,
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
        google_vision_dump: Optional[Dict[str, Any]] = None
        google_vision_err: Optional[str] = None
        if ext == ".pdf":
            if useVision:
                vision_rows, google_vision_dump, google_vision_err = _parse_pdf_with_google_vision(
                    pdf_bytes,
                    max_pages=GOOGLE_VISION_DEFAULT_MAX_PAGES if visionMaxPages is None else int(visionMaxPages),
                    dpi=int(visionDpi),
                    steps=steps,
                    include_dump=includeVisionDump,
                )
                raw_rows.extend(vision_rows)
            else:
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
            # Image upload: allow either Vision-based parse or Tesseract parse.
            if useVision:
                google_vision_dump, google_vision_err = try_google_vision_dump(pdf_bytes)
                if google_vision_dump:
                    page_data = {
                        "pageNumber": 1,
                        "width": 0.0,
                        "height": 0.0,
                        "words": _vision_dump_page_to_words(google_vision_dump),
                        "tables": [],
                        "text": _vision_dump_to_page_text(google_vision_dump),
                    }
                    page_rows = reconstruct_rows_statement_columns(page_data, source="google_vision_image")
                    if not page_rows:
                        page_rows = reconstruct_rows_geometric(page_data, source="google_vision_image")
                    raw_rows.extend(page_rows)
                    add_step(steps, "google_vision_image_parse", "success", f"rows={len(page_rows)}")
                else:
                    add_step(steps, "google_vision_image_parse", "failed", google_vision_err or "no_dump")
            else:
                # Debug-only: capture raw Google Vision response (if configured) for this image upload.
                if includeVisionDump:
                    google_vision_dump, google_vision_err = try_google_vision_dump(pdf_bytes)
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

        repaired_rows, gpt_detail, gpt_notes = repair_rows_with_gpt_compact(
            headers=["date", "description", "amount"],
            sample_rows=raw_rows[:GPT_MAX_SAMPLE_ROWS],
            failed_rows=failed_rows[:GPT_MAX_SAMPLE_ROWS],
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
        if gpt_notes:
            notes.extend(gpt_notes[:20])

        low_confidence = [t for t in transactions if float(t.get("confidence") or 0) < CONFIDENCE_REVIEW_THRESHOLD]
        if low_confidence:
            add_step(
                steps,
                "confidence_scoring",
                "success",
                f"low_confidence={len(low_confidence)} threshold={CONFIDENCE_REVIEW_THRESHOLD}",
            )
        else:
            add_step(steps, "confidence_scoring", "success", "all transactions above threshold")

        gpt_val_detail, gpt_val_notes = validate_transactions_with_gpt_compact(low_confidence or transactions)
        if gpt_val_notes:
            notes.extend(gpt_val_notes[:20])
            add_step(steps, "validate_transactions_with_gpt_compact", "success", gpt_val_detail)
        else:
            add_step(steps, "validate_transactions_with_gpt_compact", "skipped", gpt_val_detail)

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
        tx_dump = [Transaction(**t).model_dump() for t in transactions]
        return {
            "ok": True,
            "text": full_text if full_text else "",
            "textPreview": full_text[:2000] if full_text else "",
            "transactions": tx_dump,
            "statementCsv": _transactions_to_statement_csv(tx_dump),
            "googleVisionDump": google_vision_dump,
            "googleVisionDumpError": google_vision_err,
            "regexPlan": regex_plan.model_dump(),
            "steps": [Step(**s).model_dump() for s in steps],
            "notes": notes,
        }
    except Exception as exc:
        add_step(steps, "parse-bank-statement", "failed", f"Unhandled error: {exc}")
        return build_error_response("Parser error", steps, notes + [str(exc)], status_code=500)

