import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from app import main as app
from app.barcodes import decode_page_barcodes, verify_central_numbers
from app.validation import normalize_israeli_id, normalize_received_date


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"
PAGE_DIR = EVAL_DIR / "pages"
CACHE_DIR = EVAL_DIR / "cache"


def artifact_name(source_filename: str, page: int, suffix: str) -> str:
    key = f"{Path(source_filename).name.casefold()}\0{page}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()[:12]
    safe_stem = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in Path(source_filename).stem
    )[:48]
    return f"{safe_stem}-{digest}-p{page:04d}.{suffix}"


def page_path(source_filename: str, page: int) -> Path:
    return PAGE_DIR / artifact_name(source_filename, page, "jpg")


def cache_path(source_filename: str, page: int) -> Path:
    return CACHE_DIR / artifact_name(source_filename, page, "json")


def second_prompt() -> str:
    return (
        app.EXTRACTION_PROMPT
        + "\n\nThis is an independent second reading. Inspect the document from the beginning; "
        "do not rely on any prior extraction. Recheck every digit and date carefully."
    )


async def extract_raw_pair(image_bytes: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    pages = [(1, image_bytes)]
    first = await app.extract_once(pages, app.EXTRACTION_PROMPT)
    second = await app.extract_once(pages, second_prompt())
    return first, second


def read_cache(path: Path, source_filename: str, page: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        cached = json.load(source)
    if cached.get("source_filename") != Path(source_filename).name or cached.get("page") != page:
        raise ValueError(f"Cache key mismatch in {path}.")
    if not isinstance(cached.get("first"), dict) or not isinstance(cached.get("second"), dict):
        raise ValueError(f"Cache file {path} does not contain both raw extraction passes.")
    return cached


def write_cache(
    path: Path,
    source_filename: str,
    page: int,
    image_bytes: bytes,
    first: dict[str, Any],
    second: dict[str, Any],
    decoded_barcodes: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cached = {
        "source_filename": Path(source_filename).name,
        "page": page,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "first": first,
        "second": second,
        "decoded_central_barcodes": decoded_barcodes,
    }
    path.write_text(
        json.dumps(cached, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def prediction_records(
    first: dict[str, Any],
    second: dict[str, Any],
    decoded_barcodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    extracted = app.reconcile_extractions(first, second)
    id_number, id_valid = normalize_israeli_id(extracted.get("id_number"))
    received_date, date_reason = normalize_received_date(extracted.get("received_date"))
    entity = app.normalize_entity(extracted.get("referring_entity"))
    central_numbers, verification = verify_central_numbers(
        [str(number) for number in extracted.get("central_numbers", [])],
        decoded_barcodes,
    )
    extracted["id_number"] = id_number
    extracted["received_date"] = received_date
    extracted["referring_entity"] = entity
    extracted["central_numbers"] = central_numbers
    accepted = [item for item in verification if item["accepted"]]
    all_central_verified = bool(central_numbers) and (
        len(accepted) == len(central_numbers)
        and all(item["status"] == "barcode_ocr_match" for item in accepted)
    )
    if all_central_verified:
        extracted["notes"] = [
            note for note in extracted.get("notes", [])
            if not note.casefold().startswith("double extraction disagreement: central numbers")
        ]
    reasons = app.make_review_reasons(extracted, id_valid, date_reason, entity)
    if all_central_verified:
        reasons = [reason for reason in reasons if reason != "Low confidence: central numbers"]
    for item in verification:
        if item["status"] == "conflict":
            reasons.append("Central number conflict: barcode overrides the model reading")
        elif item["status"] == "barcode_only":
            reasons.append("Central number found only in barcode; confirm")
    reasons = list(dict.fromkeys(reasons))

    authorizations = extracted.get("authorizations") or [{
        "number": None,
        "confidence": 0,
        "evidence": None,
    }]
    predictions = []
    for authorization in authorizations:
        record_reasons = list(reasons)
        if not authorization.get("number"):
            record_reasons.append("Missing authorization number")
        if float(authorization.get("confidence", 0)) < app.MIN_CONFIDENCE:
            record_reasons.append("Low confidence: authorization number")
        confidence = dict(extracted["field_confidence"])
        confidence["authorization_number"] = float(authorization.get("confidence", 0))
        predictions.append({
            "first_name": extracted.get("first_name"),
            "last_name": extracted.get("last_name"),
            "id_number": id_number,
            "central_numbers": central_numbers,
            "received_date": received_date,
            "referring_entity": entity,
            "authorization_number": authorization.get("number"),
            "confidence": confidence,
            "review_reasons": list(dict.fromkeys(record_reasons)),
        })
    return predictions


def decoded_values(image_bytes: bytes) -> list[dict[str, Any]]:
    return decode_page_barcodes(image_bytes)


def load_page_image(source_filename: str, page: int, page_image: str | None = None) -> bytes:
    path = Path(page_image) if page_image else page_path(source_filename, page)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"Rendered page image not found: {path}. Run make_gt_draft.py on its source PDF first."
        )
    return path.read_bytes()
