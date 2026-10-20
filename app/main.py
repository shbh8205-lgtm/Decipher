import base64
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import httpx
import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict

from app.validation import normalize_israeli_id, normalize_received_date


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data"))).resolve()
DATABASE_PATH = DATA_DIR / "decipher.sqlite3"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PAGES = 12
MIN_CONFIDENCE = 0.75
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

EXTRACTION_PROMPT = """Read all pages of this medical payment authorization form as one document.
Return only the requested structured data. Do not infer missing or illegible values: use null,
an empty list, or a note as appropriate. Keep Hebrew names and visible number strings as written.

Field guidance:
- First and last name and ID number belong to the patient. Preserve the ID digits; never substitute
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

app = FastAPI(title="Decipher", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
                review_reasons TEXT NOT NULL,
                approved INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
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
        "review_reasons": json.loads(row["review_reasons"]),
        "approved": bool(row["approved"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


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
    payload = {
        "model": GEMINI_MODEL,
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
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured. Set GEMINI_API_KEY in the server environment.",
        )
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                GEMINI_API_URL,
                headers={"x-goog-api-key": GEMINI_API_KEY},
                json=payload,
                follow_redirects=False,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as error:
        if error.response.status_code in {401, 403}:
            detail = "Gemini rejected the API key or access to the selected model."
        elif error.response.status_code == 429:
            detail = "Gemini rate limit or quota reached. Try again later or check your API project quota."
        elif error.response.status_code == 404:
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
        else:
            detail = f"Gemini API returned HTTP {error.response.status_code}."
        raise HTTPException(status_code=502, detail=detail) from error
    except httpx.RequestError as error:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to the Gemini API. Check internet access and NetFree filtering.",
        ) from error

    try:
        result = response.json()
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
        if not content:
            raise HTTPException(status_code=502, detail="Gemini returned no text in its completed interaction.")
        extracted = json.loads(content)
    except HTTPException:
        raise
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(status_code=502, detail="Gemini returned invalid structured data.") from error
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
    return reconcile_extractions(first, second)


def save_records(filename: str, pages: list[tuple[int, bytes]], extracted: dict[str, Any]) -> list[dict[str, Any]]:
    id_number, id_valid = normalize_israeli_id(extracted.get("id_number"))
    received_date, date_reason = normalize_received_date(extracted.get("received_date"))
    entity = normalize_entity(extracted.get("referring_entity"))
    extracted["id_number"] = id_number
    extracted["received_date"] = received_date
    extracted["referring_entity"] = entity
    central_numbers = list(dict.fromkeys(str(number).strip() for number in extracted["central_numbers"] if str(number).strip()))
    authorizations = extracted["authorizations"] or [{"number": None, "confidence": 0, "evidence": None}]
    source_pages = [page_number for page_number, _ in pages]
    now = datetime.now(timezone.utc).isoformat()
    records = []

    with closing(connect_db()) as connection:
        for authorization in authorizations:
            reasons = make_review_reasons(extracted, id_valid, date_reason, entity)
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
                "review_reasons": json.dumps(list(dict.fromkeys(reasons)), ensure_ascii=False),
                "created_at": now,
                "updated_at": now,
            }
            connection.execute(
                """
                INSERT INTO authorization_records (
                    id, source_filename, source_pages, first_name, last_name, id_number,
                    central_numbers, received_date, referring_entity, authorization_number,
                    field_confidence, evidence, review_reasons, created_at, updated_at
                ) VALUES (
                    :id, :source_filename, :source_pages, :first_name, :last_name, :id_number,
                    :central_numbers, :received_date, :referring_entity, :authorization_number,
                    :field_confidence, :evidence, :review_reasons, :created_at, :updated_at
                )
                """,
                values,
            )
            records.append({**values, "source_pages": source_pages, "central_numbers": central_numbers,
                            "field_confidence": record_confidence, "evidence": record_evidence,
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


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> dict[str, Any]:
    return {
        "provider": "Gemini",
        "model": GEMINI_MODEL,
        "configured": bool(GEMINI_API_KEY),
    }


@app.get("/api/records")
def list_records() -> list[dict[str, Any]]:
    with closing(connect_db()) as connection:
        rows = connection.execute(
            "SELECT * FROM authorization_records ORDER BY created_at DESC"
        ).fetchall()
    return [record_to_dict(row) for row in rows]


@app.post("/api/extract")
async def upload_documents(
    files: Annotated[list[UploadFile], File()],
    group_images: bool = Form(False),
    external_processing_consent: bool = Form(False),
) -> dict[str, Any]:
    if not external_processing_consent:
        raise HTTPException(
            status_code=400,
            detail="Confirm that you are authorized to send these documents to Google Gemini before processing.",
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
                "documents": [{"upload_index": 0, "filename": group_name, "records": records}],
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
            extracted = await extract_document(pages)
            records = save_records(filename, pages, extracted)
            documents.append({"upload_index": index, "filename": filename, "records": records})
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
            changes["central_numbers"] = json.dumps(
                list(dict.fromkeys(value.strip() for value in (changes["central_numbers"] or []) if value.strip())),
                ensure_ascii=False,
            )
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
