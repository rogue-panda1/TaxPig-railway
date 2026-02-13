import argparse
import json
from typing import List

import requests


def format_table(rows: List[dict]) -> str:
    cols = [
        "date",
        "type",
        "amount",
        "payeePayer",
        "transactionDescription",
        "originalSourceText",
        "reference",
        "confidence",
    ]
    headers = {
        "date": "Date",
        "type": "Type",
        "amount": "Amount",
        "payeePayer": "Payee/Payer",
        "transactionDescription": "Transaction Description",
        "originalSourceText": "Original Source Text",
        "reference": "Reference",
        "confidence": "Confidence",
    }
    data = []
    for r in rows:
        row = []
        for c in cols:
            if c == "transactionDescription":
                row.append(str(r.get("transactionDescription") or r.get("description", "")))
            else:
                row.append(str(r.get(c, "")))
        data.append(row)
    widths = []
    for i, c in enumerate(cols):
        w = len(headers[c])
        for row in data:
            w = min(60, max(w, len(row[i])))
        widths.append(w)
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    out = [sep]
    out.append("| " + " | ".join(headers[c].ljust(widths[i]) for i, c in enumerate(cols)) + " |")
    out.append(sep)
    for row in data:
        clipped = [(row[i][: widths[i]]) for i in range(len(cols))]
        out.append("| " + " | ".join(clipped[i].ljust(widths[i]) for i in range(len(cols))) + " |")
    out.append(sep)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Post a statement file and print parsed transactions table.")
    ap.add_argument("--url", required=True, help="Parser endpoint, e.g. https://.../parse-bank-statement")
    ap.add_argument("--file", required=True, help="Path to statement file")
    ap.add_argument("--token", default="", help="Bearer token")
    ap.add_argument("--save-json", default="", help="Optional output path for full JSON")
    args = ap.parse_args()

    headers = {}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
    with open(args.file, "rb") as fh:
        resp = requests.post(args.url, headers=headers, files={"file": fh}, timeout=180)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("transactions", [])
    print(f"ok={payload.get('ok')} transactions={len(rows)}")
    if rows:
        print(format_table(rows))
    else:
        print("No transactions.")
    if payload.get("notes"):
        print("\nNotes:")
        for n in payload["notes"]:
            print("-", n)
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved JSON: {args.save_json}")


if __name__ == "__main__":
    main()
