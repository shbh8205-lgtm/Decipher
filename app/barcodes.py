import re
from io import BytesIO
from typing import Any, Iterator

import zxingcpp
from PIL import Image, ImageOps


def _variants(image: Image.Image, enhanced: bool) -> Iterator[tuple[Image.Image, int, int, float, float]]:
    width, height = image.size
    yield image, 0, 0, 1.0, 1.0
    if not enhanced:
        return

    grayscale = ImageOps.grayscale(image)
    contrast = ImageOps.autocontrast(grayscale)
    yield contrast, 0, 0, 1.0, 1.0
    for scale in (1.5, 2.0):
        resized = contrast.resize(
            (round(width * scale), round(height * scale)),
            Image.Resampling.LANCZOS,
        )
        yield resized, 0, 0, scale, scale

    band_height = max(1, round(height * 0.4))
    for top in (0, height - band_height):
        crop = image.crop((0, top, width, top + band_height))
        crop_contrast = ImageOps.autocontrast(ImageOps.grayscale(crop))
        yield crop_contrast, 0, top, 1.0, 1.0
        for scale in (1.5, 2.0):
            resized = crop_contrast.resize(
                (round(width * scale), round(band_height * scale)),
                Image.Resampling.LANCZOS,
            )
            yield resized, 0, top, scale, scale


def _inverse_point(
    x: float,
    y: float,
    rotation: int,
    variant_width: int,
    variant_height: int,
    offset_x: int,
    offset_y: int,
    scale_x: float,
    scale_y: float,
) -> tuple[float, float]:
    if rotation == 90:
        x, y = y, variant_height - x
    elif rotation == 180:
        x, y = variant_width - x, variant_height - y
    elif rotation == 270:
        x, y = variant_width - y, x
    return x / scale_x + offset_x, y / scale_y + offset_y


def _barcode_bbox(
    barcode: Any,
    rotation: int,
    original_width: int,
    original_height: int,
    variant_width: int,
    variant_height: int,
    offset_x: int,
    offset_y: int,
    scale_x: float,
    scale_y: float,
) -> dict[str, float]:
    position = barcode.position
    points = [
        (position.top_left.x, position.top_left.y),
        (position.top_right.x, position.top_right.y),
        (position.bottom_left.x, position.bottom_left.y),
        (position.bottom_right.x, position.bottom_right.y),
    ]
    mapped = [
        _inverse_point(
            x, y, rotation, variant_width, variant_height,
            offset_x, offset_y, scale_x, scale_y,
        )
        for x, y in points
    ]
    left = max(0.0, min(point[0] for point in mapped))
    top = max(0.0, min(point[1] for point in mapped))
    right = min(float(original_width), max(point[0] for point in mapped))
    bottom = min(float(original_height), max(point[1] for point in mapped))
    return {
        "x": left / original_width,
        "y": top / original_height,
        "width": max(0.0, right - left) / original_width,
        "height": max(0.0, bottom - top) / original_height,
    }


def _overlap(left: dict[str, float], right: dict[str, float]) -> float:
    x1 = max(left["x"], right["x"])
    y1 = max(left["y"], right["y"])
    x2 = min(left["x"] + left["width"], right["x"] + right["width"])
    y2 = min(left["y"] + left["height"], right["y"] + right["height"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        left["width"] * left["height"]
        + right["width"] * right["height"]
        - intersection
    )
    return intersection / union if union else 0.0


def _same_detection(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["text"] != right["text"]:
        return False
    if _overlap(left["bbox"], right["bbox"]) >= 0.5:
        return True
    left_box = left["bbox"]
    right_box = right["bbox"]
    left_center = (
        left_box["x"] + left_box["width"] / 2,
        left_box["y"] + left_box["height"] / 2,
    )
    right_center = (
        right_box["x"] + right_box["width"] / 2,
        right_box["y"] + right_box["height"] / 2,
    )
    return (
        abs(left_center[0] - right_center[0]) <= max(left_box["width"], right_box["width"]) * 0.2
        and abs(left_center[1] - right_center[1]) <= max(left_box["height"], right_box["height"]) * 0.5
    )


def decode_page_barcodes(image_bytes: bytes, enhanced: bool = True) -> list[dict[str, Any]]:
    with Image.open(BytesIO(image_bytes)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    original_width, original_height = image.size
    decoded: list[dict[str, Any]] = []
    for variant, offset_x, offset_y, scale_x, scale_y in _variants(image, enhanced):
        for rotation in (0, 90, 180, 270):
            rotated = variant.rotate(-rotation, expand=True) if rotation else variant
            for barcode in zxingcpp.read_barcodes(
                rotated,
                try_rotate=False,
                try_downscale=False,
                try_invert=False,
            ):
                text = barcode.text.strip()
                if not text:
                    continue
                bbox = _barcode_bbox(
                    barcode, rotation, original_width, original_height,
                    variant.width, variant.height,
                    offset_x, offset_y, scale_x, scale_y,
                )
                item = {
                    "text": text,
                    "format": str(barcode.format),
                    "rotation": rotation,
                    "bbox": bbox,
                }
                if any(_same_detection(existing, item) for existing in decoded):
                    continue
                decoded.append(item)

    return sorted(
        decoded,
        key=lambda item: (
            item["bbox"]["y"],
            item["bbox"]["x"],
            item["text"],
            item["rotation"],
        ),
    )


def verify_central_numbers(
    model_numbers: list[str],
    decoded_barcodes: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    def unique(values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    def comparable(value: str) -> str:
        return re.sub(r"[\s-]+", "", value).casefold()

    model_values = unique([str(value) for value in model_numbers])
    barcode_values = unique([
        str(item.get("text", "")).strip()
        for item in decoded_barcodes
        if re.fullmatch(r"[0-9]{6}", str(item.get("text", "")).strip())
    ])
    verifications: list[dict[str, Any]] = []

    def entry(
        number: str,
        status: str,
        barcode_value: str | None = None,
        alternatives: list[str] | None = None,
        candidates: list[str] | None = None,
        accepted: bool = True,
    ) -> dict[str, Any]:
        return {
            "number": number,
            "status": status,
            "barcode_value": barcode_value,
            "alternatives": alternatives or [],
            "barcode_candidates": candidates or [],
            "accepted": accepted,
        }

    if not barcode_values:
        return model_values, [
            entry(number, "ocr_only") for number in model_values
        ]

    model_matches = {
        comparable(model): next(
            barcode for barcode in barcode_values
            if comparable(barcode) == comparable(model)
        )
        for model in model_values
        if any(comparable(barcode) == comparable(model) for barcode in barcode_values)
    }

    if not model_values:
        if len(barcode_values) == 1:
            number = barcode_values[0]
            return [number], [entry(number, "barcode_only", number)]
        return [], [
            entry(number, "barcode_only", number, candidates=barcode_values, accepted=False)
            for number in barcode_values
        ]

    if len(barcode_values) == 1 and len(model_values) == 1:
        barcode = barcode_values[0]
        model = model_values[0]
        if comparable(barcode) == comparable(model):
            return model_values, [entry(model, "barcode_ocr_match", barcode)]
        return [barcode], [entry(
            barcode, "conflict", barcode, alternatives=[model], candidates=barcode_values
        )]

    for model in model_values:
        matched = model_matches.get(comparable(model))
        verifications.append(
            entry(
                model,
                "barcode_ocr_match" if matched else "ocr_only",
                matched,
                candidates=barcode_values,
            )
        )
    matched_barcodes = {comparable(value) for value in model_matches.values()}
    verifications.extend(
        entry(
            barcode,
            "barcode_only",
            barcode,
            candidates=barcode_values,
            accepted=False,
        )
        for barcode in barcode_values
        if comparable(barcode) not in matched_barcodes
    )
    return model_values, verifications
