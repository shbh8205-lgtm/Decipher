import argparse
import re
import sys
from io import BytesIO
from pathlib import Path

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.barcodes import decode_page_barcodes


SAMPLE_BASELINE_PAGES = 48
SAMPLE_BASELINE_ANY = 38
SAMPLE_BASELINE_SIX_DIGIT = 36


def report_pdf(path: Path) -> tuple[int, tuple[int, int], tuple[int, int]]:
    document = pymupdf.open(path)
    page_count = document.page_count
    baseline_any = baseline_six_digit = enhanced_any = enhanced_six_digit = 0
    try:
        for page in document:
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(250 / 72, 250 / 72),
                alpha=False,
            )
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            encoded = image_to_jpeg(image)
            baseline = decode_page_barcodes(encoded, enhanced=False)
            baseline_has_six_digit = any(is_six_digit(item["text"]) for item in baseline)
            enhanced = (
                baseline
                if baseline_has_six_digit
                else decode_page_barcodes(encoded, enhanced=True)
            )
            baseline_any += bool(baseline)
            baseline_six_digit += any(is_six_digit(item["text"]) for item in baseline)
            enhanced_any += bool(enhanced)
            enhanced_six_digit += any(is_six_digit(item["text"]) for item in enhanced)
    finally:
        document.close()
    return page_count, (baseline_any, baseline_six_digit), (enhanced_any, enhanced_six_digit)


def image_to_jpeg(image: Image.Image) -> bytes:
    normalized = image.convert("RGB")
    normalized.thumbnail((2400, 2400))
    output = BytesIO()
    normalized.save(output, format="JPEG", quality=88, optimize=True)
    return output.getvalue()


def is_six_digit(value: str) -> bool:
    return re.fullmatch(r"[0-9]{6}", value) is not None


def pdf_paths(inputs: list[Path]) -> list[Path]:
    paths = []
    for item in inputs:
        if item.is_dir():
            paths.extend(item.rglob("*.pdf"))
        elif item.suffix.casefold() == ".pdf":
            paths.append(item)
    return sorted(set(path.resolve() for path in paths))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure local barcode coverage on PDFs rendered at 250 DPI."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="PDF files or directories")
    args = parser.parse_args()
    paths = pdf_paths(args.paths)
    if not paths:
        parser.error("No PDF files found in the supplied paths.")

    total_pages = total_baseline_any = total_baseline_six = 0
    total_enhanced_any = total_enhanced_six = 0
    for path in paths:
        pages, baseline, enhanced = report_pdf(path)
        total_pages += pages
        total_baseline_any += baseline[0]
        total_baseline_six += baseline[1]
        total_enhanced_any += enhanced[0]
        total_enhanced_six += enhanced[1]
        print(
            f"{path.name}: {pages} pages; baseline any/six-digit "
            f"{baseline[0]}/{baseline[1]}; enhanced {enhanced[0]}/{enhanced[1]} "
            f"(+{enhanced[0] - baseline[0]}/+{enhanced[1] - baseline[1]})"
        )

    print(
        f"Total: {total_pages} pages; baseline any barcode "
        f"{total_baseline_any}/{total_pages}, six-digit numeric "
        f"{total_baseline_six}/{total_pages}; enhanced any barcode "
        f"{total_enhanced_any}/{total_pages}, six-digit numeric "
        f"{total_enhanced_six}/{total_pages}; improvement "
        f"+{total_enhanced_any - total_baseline_any}/+{total_enhanced_six - total_baseline_six} pages."
    )
    if total_pages == SAMPLE_BASELINE_PAGES:
        print(
            f"Compared with stated baseline {SAMPLE_BASELINE_ANY}/{SAMPLE_BASELINE_SIX_DIGIT}: "
            f"baseline delta {total_baseline_any - SAMPLE_BASELINE_ANY}/"
            f"{total_baseline_six - SAMPLE_BASELINE_SIX_DIGIT}; enhanced delta "
            f"{total_enhanced_any - SAMPLE_BASELINE_ANY}/"
            f"{total_enhanced_six - SAMPLE_BASELINE_SIX_DIGIT}."
        )


if __name__ == "__main__":
    main()
