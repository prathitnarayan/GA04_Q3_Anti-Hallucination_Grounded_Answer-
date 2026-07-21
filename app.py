from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from datetime import datetime

app = Flask(__name__)
CORS(app)


def parse_date(text):
    # YYYY-MM-DD
    m = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    if m:
        return m.group(1)

    # 15 March 2026
    m = re.search(r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})', text)
    if m:
        try:
            return datetime.strptime(m.group(1), "%d %B %Y").strftime("%Y-%m-%d")
        except:
            pass

    return None


def parse_amount(value):
    value = value.replace(",", "")
    nums = re.findall(r'\d+(?:\.\d+)?', value)
    if nums:
        return float(nums[-1])
    return None


@app.route("/")
def home():
    return jsonify({"status": "running"})


@app.route("/extract", methods=["POST"])
def extract():

    data = request.get_json(force=True)
    text = data.get("invoice_text", "")

    result = {
        "invoice_no": None,
        "date": None,
        "vendor": None,
        "amount": None,
        "tax": None,
        "currency": None
    }

    # Invoice number
    patterns = [
        r'Invoice No[: ]+([A-Za-z0-9\-/]+)',
        r'Ref[: ]+([A-Za-z0-9\-/]+)'
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            result["invoice_no"] = m.group(1).strip()
            break

    # Date
    result["date"] = parse_date(text)

    # Vendor
    m = re.search(r'Vendor[: ]+(.+)', text, re.I)
    if m:
        result["vendor"] = m.group(1).strip()
    else:
        lines = text.splitlines()
        if lines:
            result["vendor"] = lines[0].split("—")[0].strip()

    # Currency
    m = re.search(r'Currency[: ]+([A-Z]{3})', text)
    if m:
        result["currency"] = m.group(1)

    # Subtotal
    m = re.search(r'Subtotal.*?([\d,]+\.\d+)', text, re.I)
    if m:
        result["amount"] = parse_amount(m.group(1))

    # Tax
    m = re.search(r'(?:GST|IGST|CGST|SGST).*?([\d,]+\.\d+)', text, re.I)
    if m:
        result["tax"] = parse_amount(m.group(1))

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)