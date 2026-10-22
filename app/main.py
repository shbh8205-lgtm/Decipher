import base64
import asyncio
import csv
import json
import os
import re
import secrets
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4
from hashlib import sha256
from base64 import b64encode

import bcrypt
import httpx
import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict
from starlette.middleware.sessions import SessionMiddleware

from app.barcodes import decode_page_barcodes, verify_central_numbers
from app.validation import normalize_israeli_id, normalize_received_date


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data"))).resolve()
DATABASE_PATH = DATA_DIR / "decipher.sqlite3"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEY_BACKUP = os.getenv("GEMINI_API_KEY_BACKUP", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
SESSION_SECRET = os.getenv("SESSION_SECRET") or secrets.token_urlsafe(32)
DEMO_BANNER = os.getenv("DEMO_BANNER", "").strip().casefold() in {"1", "true", "yes", "on"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PAGES = 12
MIN_CONFIDENCE = 0.75
GEMINI_MAX_ATTEMPTS = 3
LOGIN_LIMIT = 10
LOGIN_WINDOW_SECONDS = 15 * 60
EXTRACT_LIMIT = 10
EXTRACT_WINDOW_SECONDS = 60 * 60
SESSION_MAX_AGE_SECONDS = 12 * 60 * 60
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

ENTITY_ALIASES = {
    "clalit": "Clalit",
    "כללית": "Clalit",
    "maccabi": "Maccabi",
    "מכבי": "Maccabi",
    "meuhedet": "Meuhedet",
    "מאוחדת": "Meuhedet",
    "leumit": "Leumit",
    "לאומית": "Leumit",
}

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "first_name": {"type": ["string", "null"]},
        "last_name": {"type": ["string", "null"]},
        "id_number": {"type": ["string", "null"]},
        "central_numbers": {"type": "array", "items": {"type": "string"}},
        "received_date": {"type": ["string", "null"]},
        "referring_entity": {"type": ["string", "null"]},
        "authorizations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": ["string", "null"]},
                    "label": {"type": ["string", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": ["string", "null"]},
                },
                "required": ["number", "label", "confidence", "evidence"],
                "additionalProperties": False,
            },
        },
        "field_confidence": {
            "type": "object",
            "properties": {
                "first_name": {"type": "number", "minimum": 0, "maximum": 1},
                "last_name": {"type": "number", "minimum": 0, "maximum": 1},
                "id_number": {"type": "number", "minimum": 0, "maximum": 1},
                "central_numbers": {"type": "number", "minimum": 0, "maximum": 1},
                "received_date": {"type": "number", "minimum": 0, "maximum": 1},
                "referring_entity": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": [
                "first_name",
                "last_name",
                "id_number",
                "central_numbers",
                "received_date",
                "referring_entity",
            ],
            "additionalProperties": False,
        },
        "evidence": {
            "type": "object",
            "properties": {
                "first_name": {"type": ["string", "null"]},
                "last_name": {"type": ["string", "null"]},
                "id_number": {"type": ["string", "null"]},
                "central_numbers": {"type": ["string", "null"]},
                "received_date": {"type": ["string", "null"]},
                "referring_entity": {"type": ["string", "null"]},
            },
            "required": [
                "first_name",
                "last_name",
                "id_number",
                "central_numbers",
                "received_date",
                "referring_entity",
            ],
            "additionalProperties": False,
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "first_name",
        "last_name",
        "id_number",
        "central_numbers",
        "received_date",
        "referring_entity",
        "authorizations",
        "field_confidence",
        "evidence",
        "notes",
    ],
    "additionalProperties": False,
}

EXTRACTION_PROMPT = """Read the supplied page or grouped pages of this medical payment authorization form.
Return only the requested structured data. Do not infer missing or illegible values: use null,
an empty list, or a note as appropriate. Keep Hebrew only names and visible number strings as written.

Field guidance:
- First and last name in hebrew only (when its hand writing notice to read from right to left) and ID number belong to the patient. Preserve the ID digits; never substitute
  another number from the form.
- A "Central" / "Merkazi" (מרכזי) number is usually on a distinct sticker labeled מרכזי. Do not
  confuse it with an ID, barcode, phone, or authorization number. Return every clearly identified
  central number; do not invent an association when several appear.
- Receipt date is the date beside a "Received" / "Hitkabel" (התקבל) stamp, or a handwritten date
  below/near the Central number. Do not use form, birth, signature, or print dates instead.
- The referring entity is the HMO or hospital shown by its name/logo.
- Authorization entries may be labeled authorization/commitment (התחייבות), requirement/demand
  (דרישה), or, for Leumit, sample (דגימה). Return each distinct number and its visible label.
- Return what is visible, not a guess. For each field, provide confidence from 0 to 1 and a short
  evidence quote or visual description; notes should explain ambiguity. Use one authorization
  object per distinct authorization. Do not combine unrelated numbers."""


def gemini_response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert the extraction JSON Schema to Gemini's supported responseSchema format."""
    result: dict[str, Any] = {}
    schema_type = schema.get("type")
    nullable = isinstance(schema_type, list) and "null" in schema_type
    if nullable:
        schema_type = next(item for item in schema_type if item != "null")
    if schema_type:
        result["type"] = schema_type.upper()
    if nullable:
        result["nullable"] = True
    if "properties" in schema:
        result["properties"] = {
            key: gemini_response_schema(value)
            for key, value in schema["properties"].items()
        }
    if "items" in schema:
        result["items"] = gemini_response_schema(schema["items"])
    if "required" in schema:
        result["required"] = schema["required"]
    if "minimum" in schema:
        result["minimum"] = schema["minimum"]
    if "maximum" in schema:
        result["maximum"] = schema["maximum"]
    return result

app = FastAPI(title="Decipher", version="0.1.0", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

LOGIN_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Decipher | Sign in</title><style>
body{margin:0;background:#f3f7f5;color:#18302f;font:16px system-ui,sans-serif;min-height:100vh;display:grid;place-items:center}
main{width:min(390px,calc(100% - 40px));padding:28px;background:#fff;border:1px solid #dce6e3;border-radius:14px;box-shadow:0 2px 8px #19302f08}
h1{margin:0 0 8px;font-size:24px}p{color:#617472;line-height:1.5}label{display:block;margin-top:16px;font-weight:600;font-size:13px}
input{box-sizing:border-box;width:100%;margin-top:6px;padding:11px;border:1px solid #dce6e3;border-radius:8px;font:inherit}
button{width:100%;margin-top:20px;padding:11px;border:0;border-radius:8px;background:#087f72;color:#fff;font:inherit;font-weight:650;cursor:pointer}
.error{padding:10px;color:#a33434;background:#fff0f0;border-radius:8px}
</style></head><body><main><h1>Decipher</h1><p>Sign in to the authorization review workspace.</p>
{error}<form method="post" action="/login"><label for="username">Username</label>
<input id="username" name="username" autocomplete="username" maxlength="128" required>
<label for="password">Password</label><input id="password" name="password" type="password" autocomplete="current-password" maxlength="1024" required>
<button type="submit">Sign in</button></form></main></body></html>"""


def configured_users() -> dict[str, bytes]:
    users: dict[str, bytes] = {}
    for entry in os.getenv("APP_USERS", "").split(","):
        username, separator, password_hash = entry.strip().partition(":")
        if separator and username.strip() and password_hash.startswith("$2"):
            users[username.strip()] = password_hash.encode("ascii", errors="ignore")
    single_hash = os.getenv("APP_PASSWORD_HASH", "")
    if single_hash.startswith("$2") and not users:
        users["operator"] = single_hash.encode("ascii", errors="ignore")
    return users


_DUMMY_PASSWORD_HASH = bcrypt.hashpw(
    b"decipher-invalid-user-password",
    bcrypt.gensalt(),
)
_RATE_BUCKETS: dict[tuple[str, str], tuple[float, int]] = {}


def client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def rate_limited(key: tuple[str, str], limit: int, window_seconds: int) -> bool:
    now = time.monotonic()
    expired = [
        bucket for bucket, (started, _) in _RATE_BUCKETS.items()
        if now - started >= window_seconds
    ]
    for bucket in expired:
        _RATE_BUCKETS.pop(bucket, None)
    started, count = _RATE_BUCKETS.get(key, (now, 0))
    if now - started >= window_seconds:
        started, count = now, 0
    count += 1
    _RATE_BUCKETS[key] = (started, count)
    return count > limit


def password_matches(username: str, password: str) -> bool:
    users = configured_users()
    expected_hash = users.get(username, _DUMMY_PASSWORD_HASH)
    try:
        valid = bcrypt.checkpw(password.encode("utf-8"), expected_hash)
    except (ValueError, TypeError):
        valid = bcrypt.checkpw(b"invalid-password", _DUMMY_PASSWORD_HASH)
    return username in users and valid


def security_headers(response: Response) -> Response:
    inline_hashes = []
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for script in re.findall(r"<script(?:\s[^>]*)?>(.*?)</script\s*>", html, re.IGNORECASE | re.DOTALL):
        digest = b64encode(sha256(script.encode("utf-8")).digest()).decode("ascii")
        inline_hashes.append(f"'sha256-{digest}'")
    script_sources = " ".join(["'self'", *inline_hashes])
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        f"script-src {script_sources}; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "frame-src 'self' blob:; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    )
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


@app.middleware("http")
async def protect_and_harden(request: Request, call_next: Any) -> Response:
    path = request.url.path
    is_api = path == "/api" or path.startswith("/api/")
    if path != "/healthz" and (path == "/" or is_api) and not request.session.get("username"):
        if is_api:
            response: Response = Response(
                content='{"detail":"Authentication required"}',
                status_code=401,
                media_type="application/json",
            )
        else:
            response = RedirectResponse("/login", status_code=303)
        return security_headers(response)
    if path == "/api/extract" and request.method == "POST":
        if rate_limited(
            ("extract", client_key(request)),
            EXTRACT_LIMIT,
            EXTRACT_WINDOW_SECONDS,
        ):
            return security_headers(Response(
                content='{"detail":"Rate limit exceeded. Try again later."}',
                status_code=429,
                media_type="application/json",
            ))
    return security_headers(await call_next(request))


app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="decipher_session",
    max_age=SESSION_MAX_AGE_SECONDS,
    same_site="lax",
    https_only=True,
)


def connect_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_db() -> None:
    with closing(connect_db()) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS authorization_records (
                id TEXT PRIMARY KEY,
                source_filename TEXT NOT NULL,
                source_pages TEXT NOT NULL,
                first_name TEXT,
                last_name TEXT,
                id_number TEXT,
                central_numbers TEXT NOT NULL,
                received_date TEXT,
                referring_entity TEXT,
                authorization_number TEXT,
                field_confidence TEXT NOT NULL,
                evidence TEXT NOT NULL,
                preview_thumbnail TEXT,
                barcode_verification TEXT NOT NULL DEFAULT '[]',
                review_reasons TEXT NOT NULL,
                approved INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        columns = {
            column["name"]
            for column in connection.execute("PRAGMA table_info(authorization_records)")
        }
        if "preview_thumbnail" not in columns:
            connection.execute(
                "ALTER TABLE authorization_records ADD COLUMN preview_thumbnail TEXT"
            )
        if "barcode_verification" not in columns:
            connection.execute(
                "ALTER TABLE authorization_records ADD COLUMN barcode_verification TEXT NOT NULL DEFAULT '[]'"
            )
        connection.commit()


def seed_demo_records() -> None:
    if os.getenv("DEMO_SEED", "").strip() != "1":
        return
    now = datetime.now(timezone.utc).isoformat()
    received_date = datetime.now(timezone.utc).strftime("%d/%m/%Y")
    confidence = json.dumps({
        "first_name": 1,
        "last_name": 1,
        "id_number": 1,
        "central_numbers": 1,
        "received_date": 1,
        "referring_entity": 1,
        "authorization_number": 1,
    })
    evidence = json.dumps({"demo": "Synthetic sample data only."})
    samples = [
        ("Sample", "Patient A", "930001", "DEMO-001", "Clalit"),
        ("Sample", "Patient B", "930002", "DEMO-002", "Maccabi"),
        ("Sample", "Patient C", "930003", "DEMO-003", "Meuhedet"),
    ]
    with closing(connect_db()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        if connection.execute(
            "SELECT 1 FROM authorization_records LIMIT 1"
        ).fetchone():
            connection.rollback()
            return
        for first_name, last_name, central_number, authorization_number, entity in samples:
            connection.execute(
                """
                INSERT INTO authorization_records (
                    id, source_filename, source_pages, first_name, last_name, id_number,
                    central_numbers, received_date, referring_entity, authorization_number,
                    field_confidence, evidence, preview_thumbnail, barcode_verification,
                    review_reasons, approved, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    "DEMO SAMPLE ONLY",
                    "[1]",
                    first_name,
                    last_name,
                    None,
                    json.dumps([central_number]),
                    received_date,
                    entity,
                    authorization_number,
                    confidence,
                    evidence,
                    None,
                    "[]",
                    json.dumps(["Demo seed: sample-only data"]),
                    0,
                    now,
                    now,
                ),
            )
        connection.commit()


def record_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "source_filename": row["source_filename"],
        "source_pages": json.loads(row["source_pages"]),
        "first_name": row["first_name"],
        "last_name": row["last_name"],
        "id_number": row["id_number"],
        "central_numbers": json.loads(row["central_numbers"]),
        "received_date": row["received_date"],
        "referring_entity": row["referring_entity"],
        "authorization_number": row["authorization_number"],
        "field_confidence": json.loads(row["field_confidence"]),
        "evidence": json.loads(row["evidence"]),
        "preview_thumbnail": row["preview_thumbnail"],
        "barcode_verification": json.loads(row["barcode_verification"]),
        "review_reasons": json.loads(row["review_reasons"]),
        "approved": bool(row["approved"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def csv_text_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + text
    return text


def records_csv(records: list[dict[str, Any]]) -> str:
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerow([
        "First Name",
        "Last Name",
        "ID Number (text)",
        "Central Number(s)",
        "Sample Receipt Date",
        "Referring Entity",
        "Authorization Number",
        "Central Verification",
        "Requires Review",
        "Review Reasons",
        "Approved",
        "Source Filename",
        "Source Page(s)",
    ])
    for record in records:
        id_number = record["id_number"] or ""
        # Excel otherwise drops leading zeroes from numeric-looking ID values.
        excel_text_id = f'="{id_number.replace(chr(34), chr(34) * 2)}"' if id_number else ""
        writer.writerow([
            csv_text_cell(record["first_name"]),
            csv_text_cell(record["last_name"]),
            excel_text_id,
            csv_text_cell("; ".join(record["central_numbers"])),
            csv_text_cell(record["received_date"]),
            csv_text_cell(record["referring_entity"]),
            csv_text_cell(record["authorization_number"]),
            csv_text_cell("; ".join(
                f'{item["number"]}: {item["status"]}'
                for item in record["barcode_verification"]
            )),
            "Yes" if record["review_reasons"] else "No",
            csv_text_cell("; ".join(record["review_reasons"])),
            "Yes" if record["approved"] else "No",
            csv_text_cell(record["source_filename"]),
            csv_text_cell(", ".join(str(page) for page in record["source_pages"])),
        ])
    return "\ufeff" + output.getvalue()


def normalize_entity(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return ENTITY_ALIASES.get(stripped.casefold(), ENTITY_ALIASES.get(stripped, stripped or None))


def make_review_reasons(
    extracted: dict[str, Any], id_valid: bool, date_reason: str | None, entity: str | None
) -> list[str]:
    reasons = list(extracted.get("notes") or [])
    confidence = extracted["field_confidence"]
    visible_fields = {
        "first_name": extracted["first_name"],
        "last_name": extracted["last_name"],
        "id_number": extracted["id_number"],
        "central_numbers": extracted["central_numbers"],
        "received_date": extracted["received_date"],
        "referring_entity": entity,
    }
    for field, value in visible_fields.items():
        if not value:
            reasons.append(f"Missing {field.replace('_', ' ')}")
        if float(confidence.get(field, 0)) < MIN_CONFIDENCE:
            reasons.append(f"Low confidence: {field.replace('_', ' ')}")
    if extracted["id_number"] and not id_valid:
        reasons.append("Invalid Israeli ID check digit or length")
    if date_reason:
        reasons.append(date_reason)
    if entity and entity not in {"Clalit", "Maccabi", "Meuhedet", "Leumit"}:
        reasons.append("Unrecognized referring entity; confirm whether this is a hospital")
    return list(dict.fromkeys(reasons))


def image_to_jpeg(image: Image.Image) -> bytes:
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    normalized.thumbnail((2400, 2400))
    output = BytesIO()
    normalized.save(output, format="JPEG", quality=88, optimize=True)
    return output.getvalue()


def make_preview_thumbnail(image_bytes: bytes) -> str:
    with Image.open(BytesIO(image_bytes)) as image:
        thumbnail = ImageOps.exif_transpose(image).convert("RGB")
        thumbnail.thumbnail((240, 300))
        output = BytesIO()
        thumbnail.save(output, format="JPEG", quality=72, optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


def normalize_document(filename: str, content: bytes) -> list[tuple[int, bytes]]:
    extension = Path(filename).suffix.casefold()
    pages: list[tuple[int, bytes]] = []
    if extension == ".pdf":
        try:
            document = pymupdf.open(stream=content, filetype="pdf")
        except Exception as error:
            raise ValueError("Could not read this PDF.") from error
        page_count = document.page_count
        if page_count > MAX_PAGES:
            document.close()
            raise ValueError(f"PDF has {page_count} pages; the limit is {MAX_PAGES}.")
        try:
            for index, page in enumerate(document):
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(250 / 72, 250 / 72), alpha=False)
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                pages.append((index + 1, image_to_jpeg(image)))
        except Exception as error:
            raise ValueError("Could not render a page in this PDF.") from error
        finally:
            document.close()
        return pages

    try:
        with Image.open(BytesIO(content)) as image:
            frame_count = getattr(image, "n_frames", 1)
            if frame_count > MAX_PAGES:
                raise ValueError(f"Image has {frame_count} frames; the limit is {MAX_PAGES}.")
            for index in range(frame_count):
                image.seek(index)
                pages.append((index + 1, image_to_jpeg(image.copy())))
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Could not read this image.") from error
    return pages


async def extract_once(pages: list[tuple[int, bytes]], prompt: str) -> dict[str, Any]:
    providers = []
    if GEMINI_API_KEY.strip():
        providers.append(("Gemini", GEMINI_API_URL, GEMINI_API_KEY.strip(), GEMINI_MODEL))
    if GEMINI_API_KEY_BACKUP.strip() and GEMINI_API_KEY_BACKUP.strip() != GEMINI_API_KEY.strip():
        providers.append(("Gemini", GEMINI_API_URL, GEMINI_API_KEY_BACKUP.strip(), GEMINI_MODEL))
    if OPENROUTER_API_KEY.strip():
        providers.append(("OpenRouter", OPENROUTER_API_URL, OPENROUTER_API_KEY.strip(), OPENROUTER_MODEL))
    if not providers:
        raise HTTPException(
            status_code=503,
            detail="No model API key is configured. Set GEMINI_API_KEY, GEMINI_API_KEY_BACKUP, or OPENROUTER_API_KEY in the server environment.",
        )
    provider_name = ""
    model_name = ""
    response: httpx.Response
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            for provider_index, (provider_name, api_url, api_key, model_name) in enumerate(providers):
                request_headers = (
                    {"x-goog-api-key": api_key}
                    if provider_name == "Gemini"
                    else {"Authorization": f"Bearer {api_key}"}
                )
                if provider_name == "Gemini":
                    payload = {
                        "model": model_name,
                        "input": [
                            {"type": "text", "text": prompt},
                            *[
                                {
                                    "type": "image",
                                    "data": base64.b64encode(content).decode("ascii"),
                                    "mime_type": "image/jpeg",
                                }
                                for _, content in pages
                            ],
                        ],
                        "response_format": {
                            "type": "text",
                            "mime_type": "application/json",
                            "schema": gemini_response_schema(EXTRACTION_SCHEMA),
                        },
                        "store": False,
                    }
                else:
                    payload = {
                        "model": model_name,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                *[
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": (
                                                "data:image/jpeg;base64,"
                                                f"{base64.b64encode(content).decode('ascii')}"
                                            )
                                        },
                                    }
                                    for _, content in pages
                                ],
                            ],
                        }],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "authorization_extraction",
                                "strict": True,
                                "schema": EXTRACTION_SCHEMA,
                            },
                        },
                    }
                use_backup = False
                for attempt in range(GEMINI_MAX_ATTEMPTS):
                    try:
                        response = await client.post(
                            api_url,
                            headers=request_headers,
                            json=payload,
                            follow_redirects=False,
                        )
                        response.raise_for_status()
                    except httpx.HTTPStatusError as error:
                        if error.response.status_code == 503 and attempt + 1 < GEMINI_MAX_ATTEMPTS:
                            await asyncio.sleep(attempt + 1)
                            continue
                        if (
                            error.response.status_code in {401, 403, 429, 503}
                            and provider_index + 1 < len(providers)
                        ):
                            use_backup = True
                            break
                        raise
                    except httpx.RequestError:
                        if provider_index + 1 < len(providers):
                            use_backup = True
                            break
                        raise
                    else:
                        break
                if not use_backup:
                    break
    except httpx.HTTPStatusError as error:
        if provider_name == "OpenRouter" and error.response.status_code in {401, 403}:
            detail = "OpenRouter rejected the API key or access to the selected model."
        elif provider_name == "Gemini" and error.response.status_code in {401, 403}:
            detail = "Gemini rejected the API key or access to the selected model."
        elif error.response.status_code == 429:
            detail = f"{provider_name} rate limit or quota reached. Try again later or check your account quota."
        elif error.response.status_code == 503:
            detail = (
                f"{provider_name} remained unavailable (HTTP 503) after {GEMINI_MAX_ATTEMPTS} attempts. "
                "Wait a few minutes and try again; if it continues, check the provider's service status "
                "or whether network filtering is interrupting the connection."
            )
        elif provider_name == "Gemini" and error.response.status_code == 404:
            api_message = ""
            try:
                api_message = str(error.response.json().get("error", {}).get("message", ""))
            except (ValueError, AttributeError):
                pass
            detail = (
                f"Gemini returned 404 for model '{GEMINI_MODEL}'. Check that GEMINI_MODEL is an "
                "available model ID and that NetFree is not "
                "replacing or blocking the Gemini API response."
            )
            if api_message:
                detail = f"{detail} Gemini message: {api_message[:300]}"
        elif provider_name == "OpenRouter" and error.response.status_code == 404:
            detail = f"OpenRouter returned 404 for model '{model_name}'. Check OPENROUTER_MODEL."
        else:
            detail = f"{provider_name} API returned HTTP {error.response.status_code}."
        raise HTTPException(status_code=502, detail=detail) from error
    except httpx.RequestError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to {provider_name}. Check internet access and network filtering.",
        ) from error

    try:
        result = response.json()
        if provider_name == "Gemini":
            if result.get("status") not in {None, "completed"}:
                raise HTTPException(
                    status_code=502,
                    detail=f"Gemini interaction did not complete (status: {result['status']}).",
                )
            content = "".join(
                item["text"]
                for step in result["steps"]
                if step.get("type") == "model_output"
                for item in step.get("content", [])
                if item.get("type") == "text" and "text" in item
            )
        else:
            message = result["choices"][0]["message"]
            if message.get("refusal"):
                raise HTTPException(status_code=502, detail="OpenRouter model refused the extraction request.")
            content = message["content"]
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") for item in content if item.get("type") == "text"
                )
        if not content:
            raise HTTPException(status_code=502, detail=f"{provider_name} returned no text.")
        extracted = json.loads(content)
    except HTTPException:
        raise
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(status_code=502, detail=f"{provider_name} returned invalid structured data.") from error
    return extracted


def reconcile_extractions(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    reconciled = dict(first)
    confidence = dict(first["field_confidence"])
    evidence = dict(first["evidence"])
    disagreements = []
    string_fields = ("first_name", "last_name", "id_number", "received_date", "referring_entity")

    def comparable(field: str, value: Any) -> str:
        if field == "id_number":
            normalized, _ = normalize_israeli_id(str(value or ""))
            return normalized or ""
        if field == "received_date":
            normalized, _ = normalize_received_date(str(value or ""))
            if normalized:
                return normalized
        if field == "referring_entity":
            return (normalize_entity(str(value)) or "").casefold()
        return " ".join(str(value or "").split()).casefold()

    for field in string_fields:
        first_value = first.get(field)
        second_value = second.get(field)
        if comparable(field, first_value) != comparable(field, second_value):
            disagreements.append(field.replace("_", " "))
            confidence[field] = 0
            reconciled[field] = first_value or second_value
        else:
            confidence[field] = min(
                float(first["field_confidence"].get(field, 0)),
                float(second["field_confidence"].get(field, 0)),
            )
        evidence[field] = first["evidence"].get(field) or second["evidence"].get(field)

    first_central = first.get("central_numbers") or []
    second_central = second.get("central_numbers") or []
    central_key = lambda value: " ".join(str(value).split()).casefold()
    if {central_key(value) for value in first_central} != {central_key(value) for value in second_central}:
        disagreements.append("central numbers")
        confidence["central_numbers"] = 0
    else:
        confidence["central_numbers"] = min(
            float(first["field_confidence"].get("central_numbers", 0)),
            float(second["field_confidence"].get("central_numbers", 0)),
        )
    reconciled["central_numbers"] = list(
        dict.fromkeys([*first_central, *second_central])
    )
    evidence["central_numbers"] = first["evidence"].get("central_numbers") or second["evidence"].get("central_numbers")
    reconciled["field_confidence"] = confidence
    reconciled["evidence"] = evidence

    auth_key = lambda authorization: " ".join(str(authorization.get("number") or "").split()).casefold()
    first_authorizations = {auth_key(item): item for item in first.get("authorizations", [])}
    second_authorizations = {auth_key(item): item for item in second.get("authorizations", [])}
    first_keys = set(first_authorizations)
    second_keys = set(second_authorizations)
    if first_keys != second_keys:
        disagreements.append("authorization numbers")
    merged_authorizations = []
    for key in dict.fromkeys([*first_authorizations, *second_authorizations]):
        first_item = first_authorizations.get(key)
        second_item = second_authorizations.get(key)
        chosen = dict(first_item or second_item)
        chosen["confidence"] = (
            min(float(first_item["confidence"]), float(second_item["confidence"]))
            if first_item and second_item and first_keys == second_keys
            else 0
        )
        if first_item and second_item and first_item.get("label") != second_item.get("label"):
            disagreements.append("authorization labels")
            chosen["confidence"] = 0
        chosen["evidence"] = (first_item or {}).get("evidence") or (second_item or {}).get("evidence")
        merged_authorizations.append(chosen)
    reconciled["authorizations"] = merged_authorizations
    reconciled["notes"] = list(dict.fromkeys([
        *(first.get("notes") or []),
        *(second.get("notes") or []),
        *[f"Double extraction disagreement: {field}" for field in disagreements],
    ]))
    return reconciled


async def extract_document(pages: list[tuple[int, bytes]]) -> dict[str, Any]:
    second_prompt = (
        EXTRACTION_PROMPT
        + "\n\nThis is an independent second reading. Inspect the document from the beginning; "
        "do not rely on any prior extraction. Recheck every digit and date carefully."
    )
    first = await extract_once(pages, EXTRACTION_PROMPT)
    second = await extract_once(pages, second_prompt)
    reconciled = reconcile_extractions(first, second)
    decoded_pages = []
    for _, content in pages:
        decoded_pages.append(await asyncio.to_thread(decode_page_barcodes, content))
    reconciled["decoded_central_barcodes"] = [
        barcode for page_barcodes in decoded_pages for barcode in page_barcodes
    ]
    return reconciled


def save_records(filename: str, pages: list[tuple[int, bytes]], extracted: dict[str, Any]) -> list[dict[str, Any]]:
    id_number, id_valid = normalize_israeli_id(extracted.get("id_number"))
    received_date, date_reason = normalize_received_date(extracted.get("received_date"))
    entity = normalize_entity(extracted.get("referring_entity"))
    extracted["id_number"] = id_number
    extracted["received_date"] = received_date
    extracted["referring_entity"] = entity
    central_numbers, barcode_verification = verify_central_numbers(
        [str(number) for number in extracted["central_numbers"]],
        extracted.get("decoded_central_barcodes", []),
    )
    extracted["central_numbers"] = central_numbers
    preview_thumbnail = make_preview_thumbnail(pages[0][1]) if pages else None
    authorizations = extracted["authorizations"] or [{"number": None, "confidence": 0, "evidence": None}]
    source_pages = [page_number for page_number, _ in pages]
    now = datetime.now(timezone.utc).isoformat()
    records = []

    with closing(connect_db()) as connection:
        for authorization in authorizations:
            accepted_verification = [
                item for item in barcode_verification if item["accepted"]
            ]
            all_central_numbers_verified = bool(central_numbers) and (
                len(accepted_verification) == len(central_numbers)
                and all(item["status"] == "barcode_ocr_match" for item in accepted_verification)
            )
            review_extracted = dict(extracted)
            if all_central_numbers_verified:
                review_extracted["notes"] = [
                    note for note in extracted.get("notes", [])
                    if not note.casefold().startswith("double extraction disagreement: central numbers")
                ]
            reasons = make_review_reasons(review_extracted, id_valid, date_reason, entity)
            if all_central_numbers_verified:
                reasons = [reason for reason in reasons if reason != "Low confidence: central numbers"]
            for verification in barcode_verification:
                if verification["status"] == "conflict":
                    reasons.append("Central number conflict: barcode overrides the model reading")
                elif verification["status"] == "barcode_only":
                    reasons.append("Central number found only in barcode; confirm")
            if not authorization.get("number"):
                reasons.append("Missing authorization number")
            if float(authorization.get("confidence", 0)) < MIN_CONFIDENCE:
                reasons.append("Low confidence: authorization number")
            record_id = str(uuid4())
            record_confidence = {
                **extracted["field_confidence"],
                "authorization_number": float(authorization.get("confidence", 0)),
            }
            record_evidence = {
                **extracted["evidence"],
                "authorization_number": authorization.get("evidence"),
            }
            values = {
                "id": record_id,
                "source_filename": Path(filename).name,
                "source_pages": json.dumps(source_pages),
                "first_name": extracted.get("first_name"),
                "last_name": extracted.get("last_name"),
                "id_number": id_number,
                "central_numbers": json.dumps(central_numbers, ensure_ascii=False),
                "received_date": received_date,
                "referring_entity": entity,
                "authorization_number": authorization.get("number"),
                "field_confidence": json.dumps(record_confidence),
                "evidence": json.dumps(record_evidence, ensure_ascii=False),
                "preview_thumbnail": preview_thumbnail,
                "barcode_verification": json.dumps(barcode_verification, ensure_ascii=False),
                "review_reasons": json.dumps(list(dict.fromkeys(reasons)), ensure_ascii=False),
                "created_at": now,
                "updated_at": now,
            }
            connection.execute(
                """
                INSERT INTO authorization_records (
                    id, source_filename, source_pages, first_name, last_name, id_number,
                    central_numbers, received_date, referring_entity, authorization_number,
                    field_confidence, evidence, preview_thumbnail, barcode_verification,
                    review_reasons, created_at, updated_at
                ) VALUES (
                    :id, :source_filename, :source_pages, :first_name, :last_name, :id_number,
                    :central_numbers, :received_date, :referring_entity, :authorization_number,
                    :field_confidence, :evidence, :preview_thumbnail, :barcode_verification,
                    :review_reasons, :created_at, :updated_at
                )
                """,
                values,
            )
            records.append({**values, "source_pages": source_pages, "central_numbers": central_numbers,
                            "field_confidence": record_confidence, "evidence": record_evidence,
                            "preview_thumbnail": preview_thumbnail,
                            "barcode_verification": barcode_verification,
                            "review_reasons": list(dict.fromkeys(reasons)), "approved": False})
        connection.commit()
    return records


class RecordCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_name: str | None = None
    last_name: str | None = None
    id_number: str | None = None
    central_numbers: list[str] | None = None
    received_date: str | None = None
    referring_entity: str | None = None
    authorization_number: str | None = None
    approved: bool | None = None


@app.on_event("startup")
def startup() -> None:
    initialize_db()
    seed_demo_records()


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.get("/")
def home() -> HTMLResponse:
    content = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    banner = (
        '<div class="demo-banner" role="status">Demo environment - sample data only</div>'
        if DEMO_BANNER else ""
    )
    content = content.replace("<!-- DEMO_BANNER -->", banner)
    return HTMLResponse(content)

@app.get("/login")
def login_page() -> HTMLResponse:
    return HTMLResponse(LOGIN_HTML.replace("{error}", ""))


@app.post("/login")
def login(
    request: Request,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
) -> Response:
    username = username.strip()[:128]
    password = password[:1024]
    key = ("login", client_key(request))
    if rate_limited(key, LOGIN_LIMIT, LOGIN_WINDOW_SECONDS):
        return HTMLResponse(
            LOGIN_HTML.replace(
                "{error}",
                '<p class="error">Too many attempts. Try again later.</p>',
            ),
            status_code=429,
        )
    if not password_matches(username, password):
        return HTMLResponse(
            LOGIN_HTML.replace(
                "{error}",
                '<p class="error">Invalid username or password.</p>',
            ),
            status_code=401,
        )
    request.session.clear()
    request.session["username"] = username
    return RedirectResponse("/", status_code=303)


@app.post("/logout")
def logout(request: Request) -> Response:
    request.session.clear()
    return RedirectResponse("/login", status_code=303)

@app.get("/api/status")
def status() -> dict[str, Any]:
    providers = []
    if GEMINI_API_KEY:
        providers.append(f"Gemini ({GEMINI_MODEL})")
    if GEMINI_API_KEY_BACKUP and GEMINI_API_KEY_BACKUP != GEMINI_API_KEY:
        role = "fallback" if GEMINI_API_KEY else "primary"
        providers.append(f"Gemini {role} ({GEMINI_MODEL})")
    if OPENROUTER_API_KEY:
        role = "fallback" if GEMINI_API_KEY or GEMINI_API_KEY_BACKUP else "primary"
        providers.append(f"OpenRouter {role} ({OPENROUTER_MODEL})")
    return {
        "provider": " + ".join(providers),
        "model": GEMINI_MODEL if GEMINI_API_KEY or GEMINI_API_KEY_BACKUP else OPENROUTER_MODEL,
        "configured": bool(GEMINI_API_KEY or GEMINI_API_KEY_BACKUP or OPENROUTER_API_KEY),
    }


@app.get("/api/records")
def list_records() -> list[dict[str, Any]]:
    with closing(connect_db()) as connection:
        rows = connection.execute(
            "SELECT * FROM authorization_records ORDER BY created_at DESC"
        ).fetchall()
    return [record_to_dict(row) for row in rows]


@app.get("/api/records.csv")
def export_records_csv() -> Response:
    records = list_records()
    return Response(
        content=records_csv(records),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="decipher-authorizations.csv"'},
    )


@app.delete("/api/records/{record_id}", status_code=204)
def delete_record(record_id: str) -> Response:
    with closing(connect_db()) as connection:
        cursor = connection.execute(
            "DELETE FROM authorization_records WHERE id = ?", (record_id,)
        )
        connection.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Record not found.")
    return Response(status_code=204)


@app.post("/api/extract")
async def upload_documents(
    files: Annotated[list[UploadFile], File()],
    group_images: bool = Form(False),
    external_processing_consent: bool = Form(False),
) -> dict[str, Any]:
    if not external_processing_consent:
        raise HTTPException(
            status_code=400,
            detail="Confirm that you are authorized to send these documents to the configured AI providers before processing.",
        )
    if group_images:
        filenames = [Path(upload.filename or "upload").name for upload in files]
        group_name = " + ".join(filenames)
        try:
            if not files:
                raise ValueError("Choose at least one image page to group.")
            if len(files) > MAX_PAGES:
                raise ValueError(f"A grouped form can have at most {MAX_PAGES} pages.")
            pages = []
            total_bytes = 0
            for upload, filename in zip(files, filenames):
                if Path(filename).suffix.casefold() not in ALLOWED_EXTENSIONS - {".pdf"}:
                    raise ValueError(f"{filename}: grouped forms require supported image files.")
                content = await upload.read(MAX_UPLOAD_BYTES + 1)
                total_bytes += len(content)
                if len(content) > MAX_UPLOAD_BYTES or total_bytes > MAX_UPLOAD_BYTES:
                    raise ValueError("Grouped form exceeds the 20 MB total size limit.")
                normalized_pages = normalize_document(filename, content)
                pages.extend((len(pages) + 1, image) for _, image in normalized_pages)
                if len(pages) > MAX_PAGES:
                    raise ValueError(f"A grouped form can have at most {MAX_PAGES} pages.")
            extracted = await extract_document(pages)
            records = save_records(group_name, pages, extracted)
            return {
                "documents": [{
                    "upload_index": 0,
                    "filename": group_name,
                    "records": records,
                    "preview_thumbnail": records[0]["preview_thumbnail"],
                }],
                "errors": [],
            }
        except HTTPException as error:
            return {"documents": [], "errors": [{"filename": group_name, "error": error.detail}]}
        except (ValueError, KeyError, TypeError) as error:
            return {"documents": [], "errors": [{"filename": group_name, "error": str(error)}]}

    documents = []
    errors = []
    for index, upload in enumerate(files):
        filename = Path(upload.filename or "upload").name
        if Path(filename).suffix.casefold() not in ALLOWED_EXTENSIONS:
            errors.append({"filename": filename, "error": "Unsupported file type."})
            continue
        content = await upload.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            errors.append({"filename": filename, "error": "File exceeds the 20 MB limit."})
            continue
        try:
            pages = normalize_document(filename, content)
            if not pages:
                raise ValueError("No readable pages found.")
            if Path(filename).suffix.casefold() == ".pdf":
                records = []
                page_errors = []
                for page_number, page_image in pages:
                    try:
                        extracted = await extract_document([(page_number, page_image)])
                        records.extend(
                            save_records(filename, [(page_number, page_image)], extracted)
                        )
                    except HTTPException as error:
                        page_errors.append({
                            "filename": filename,
                            "error": f"Page {page_number}: {error.detail}",
                        })
                    except (ValueError, KeyError, TypeError) as error:
                        page_errors.append({
                            "filename": filename,
                            "error": f"Page {page_number}: {error}",
                        })
                errors.extend(page_errors)
                if records:
                    documents.append({
                        "upload_index": index,
                        "filename": filename,
                        "records": records,
                        "preview_thumbnail": records[0]["preview_thumbnail"],
                    })
                elif not page_errors:
                    errors.append({
                        "filename": filename,
                        "error": "No records were extracted from this PDF.",
                    })
                continue

            extracted = await extract_document(pages)
            records = save_records(filename, pages, extracted)
            documents.append({
                "upload_index": index,
                "filename": filename,
                "records": records,
                "preview_thumbnail": records[0]["preview_thumbnail"],
            })
        except HTTPException as error:
            errors.append({"filename": filename, "error": error.detail})
        except (ValueError, KeyError, TypeError) as error:
            errors.append({"filename": filename, "error": str(error)})
    return {"documents": documents, "errors": errors}


@app.patch("/api/records/{record_id}")
def update_record(record_id: str, correction: RecordCorrection) -> dict[str, Any]:
    changes = correction.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(status_code=400, detail="No changes supplied.")

    with closing(connect_db()) as connection:
        row = connection.execute(
            "SELECT * FROM authorization_records WHERE id = ?", (record_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Record not found.")

        record = record_to_dict(row)
        reasons = list(record["review_reasons"])
        if "id_number" in changes:
            normalized_id, valid = normalize_israeli_id(changes["id_number"])
            changes["id_number"] = normalized_id
            reasons = [reason for reason in reasons if reason != "Invalid Israeli ID check digit or length"]
            if normalized_id and not valid:
                reasons.append("Invalid Israeli ID check digit or length")
        if "received_date" in changes:
            normalized_date, date_reason = normalize_received_date(changes["received_date"])
            changes["received_date"] = normalized_date
            reasons = [
                reason for reason in reasons
                if not reason.startswith("Sample receipt date")
                and reason not in {"Unrecognized sample receipt date", "Missing sample receipt date"}
            ]
            if date_reason:
                reasons.append(date_reason)
        if "referring_entity" in changes:
            changes["referring_entity"] = normalize_entity(changes["referring_entity"])
        if "central_numbers" in changes:
            corrected_central_numbers = list(dict.fromkeys(
                value.strip() for value in (changes["central_numbers"] or []) if value.strip()
            ))
            if corrected_central_numbers != record["central_numbers"]:
                changes["barcode_verification"] = json.dumps([
                    {
                        "number": number,
                        "status": "manual",
                        "barcode_value": None,
                        "alternatives": [],
                        "barcode_candidates": [],
                        "accepted": True,
                    }
                    for number in corrected_central_numbers
                ], ensure_ascii=False)
                reasons = [
                    reason for reason in reasons
                    if not reason.startswith("Central number conflict:")
                    and not reason.startswith("Central number found only in barcode")
                ]
            changes["central_numbers"] = json.dumps(corrected_central_numbers, ensure_ascii=False)
        for field in ("first_name", "last_name", "id_number", "central_numbers", "received_date",
                      "referring_entity", "authorization_number"):
            if field in changes:
                reasons = [reason for reason in reasons if reason != f"Missing {field.replace('_', ' ')}"]
                value = changes[field]
                empty = value is None or value == [] or value == ""
                if field == "central_numbers" and isinstance(value, str):
                    empty = not json.loads(value)
                if empty:
                    reasons.append(f"Missing {field.replace('_', ' ')}")
        if "referring_entity" in correction.model_fields_set:
            reasons = [
                reason for reason in reasons
                if reason != "Unrecognized referring entity; confirm whether this is a hospital"
            ]
            if changes["referring_entity"] and changes["referring_entity"] not in {
                "Clalit", "Maccabi", "Meuhedet", "Leumit"
            }:
                reasons.append("Unrecognized referring entity; confirm whether this is a hospital")
        requested_approval = changes.pop("approved", None)
        changed_fields = set(changes) - {"review_reasons", "updated_at"}
        approved = requested_approval if requested_approval is not None else (
            False if changed_fields else record["approved"]
        )
        changes["review_reasons"] = json.dumps(list(dict.fromkeys(reasons)), ensure_ascii=False)
        changes["approved"] = int(bool(approved))
        changes["updated_at"] = datetime.now(timezone.utc).isoformat()
        assignments = ", ".join(f"{column} = ?" for column in changes)
        connection.execute(
            f"UPDATE authorization_records SET {assignments} WHERE id = ?",
            [*changes.values(), record_id],
        )
        connection.commit()
        updated = connection.execute(
            "SELECT * FROM authorization_records WHERE id = ?", (record_id,)
        ).fetchone()
    return record_to_dict(updated)
