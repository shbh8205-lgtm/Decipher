import argparse
import asyncio
import csv
import hashlib
import sys
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import main as app
from scripts.evaluation_pipeline import (
    CACHE_DIR,
    PAGE_DIR,
    artifact_name,
    decoded_values,
    extract_raw_pair,
    page_path,
    prediction_records,
    read_cache,
    write_cache,
)
from scripts.evaluation_utils import GROUND_TRUTH_FIELDS


def pdf_files(inputs: list[Path]) -> list[Path]:
    files = []
    for item in inputs:
        if item.is_dir():
            files.extend(path for path in item.rglob("*") if path.is_file() and path.suffix.casefold() == ".pdf")
        elif item.is_file() and item.suffix.casefold() == ".pdf":
            files.append(item)
        else:
            raise ValueError(f"Input is not a PDF file or directory: {item}")
    unique = sorted(set(path.resolve() for path in files))
    if not unique:
        raise ValueError("No PDF files were found in the supplied input.")
    names = [path.name.casefold() for path in unique]
    if len(names) != len(set(names)):
        raise ValueError("PDF input contains duplicate filenames; source_filename must be unique.")
    return unique


def render_pdf(path: Path) -> list[tuple[int, bytes]]:
    document = pymupdf.open(path)
    pages = []
    try:
        for page_index, page in enumerate(document, 1):
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(250 / 72, 250 / 72),
                alpha=False,
            )
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            pages.append((page_index, app.image_to_jpeg(image)))
    finally:
        document.close()
    return pages


def cached_raw(
    source_filename: str,
    page: int,
    image_bytes: bytes,
) -> dict[str, Any] | None:
    path = CACHE_DIR / artifact_name(source_filename, page, "json")
    if not path.is_file():
        return None
    cached = read_cache(path, source_filename, page)
    digest = hashlib.sha256(image_bytes).hexdigest()
    return cached if cached.get("image_sha256") == digest else None


def draft_rows(
    filename: str,
    page: int,
    predictions: list[dict[str, Any]],
    split: str,
) -> list[dict[str, str]]:
    if not predictions:
        predictions = [{}]
    result = []
    for prediction in predictions:
        result.append({
            "source_filename": filename,
            "page": str(page),
            "first_name": prediction.get("first_name") or "",
            "last_name": prediction.get("last_name") or "",
            "id_number": prediction.get("id_number") or "",
            "central_numbers": ";".join(prediction.get("central_numbers", [])),
            "received_date": prediction.get("received_date") or "",
            "referring_entity": prediction.get("referring_entity") or "",
            "authorization_number": prediction.get("authorization_number") or "",
            "split": split,
            "notes": "",
            "page_image": str(page_path(filename, page).relative_to(ROOT)),
        })
    return result


async def create_draft(inputs: list[Path], split: str, output: Path) -> int:
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    page_count = 0
    for pdf in pdf_files(inputs):
        for page, image_bytes in render_pdf(pdf):
            page_count += 1
            filename = pdf.name
            image_path = page_path(filename, page)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(image_bytes)
            cached = cached_raw(filename, page, image_bytes)
            if cached is None:
                first, second = await extract_raw_pair(image_bytes)
                barcodes = await asyncio.to_thread(decoded_values, image_bytes)
                write_cache(
                    CACHE_DIR / artifact_name(filename, page, "json"),
                    filename,
                    page,
                    image_bytes,
                    first,
                    second,
                    barcodes,
                )
            else:
                first = cached["first"]
                second = cached["second"]
                barcodes = cached.get("decoded_central_barcodes", [])
            predictions = prediction_records(first, second, barcodes)
            rows.extend(draft_rows(filename, page, predictions, split))
            print(f"Prepared {filename}, page {page}.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as target:
        target.write(
            "# DRAFT ONLY: model values are prefilled predictions, not ground truth. "
            "A human must verify every field against page_image before use.\r\n"
        )
        columns = [*GROUND_TRUTH_FIELDS, "page_image"]
        writer = csv.DictWriter(target, fieldnames=columns, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} draft authorization row(s) from {page_count} page(s) to {output}.")
    print("Verify every value against the rendered page before treating any row as ground truth.")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a manually verified ground-truth CSV draft from PDFs."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="PDF files or folders containing PDFs")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval" / "ground_truth_draft.csv",
    )
    args = parser.parse_args()
    try:
        asyncio.run(create_draft(args.paths, args.split, args.output.resolve()))
    except Exception as error:
        print(f"Draft generation failed: {type(error).__name__}: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
