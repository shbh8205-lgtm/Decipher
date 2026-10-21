import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.main import MIN_CONFIDENCE, normalize_entity
from app.validation import normalize_received_date


GROUND_TRUTH_FIELDS = (
    "source_filename",
    "page",
    "first_name",
    "last_name",
    "id_number",
    "central_numbers",
    "received_date",
    "referring_entity",
    "authorization_number",
    "split",
    "notes",
)
SCALAR_FIELDS = (
    "first_name",
    "last_name",
    "id_number",
    "received_date",
    "referring_entity",
)
SET_FIELDS = ("central_numbers", "authorization_number")
EVALUATED_FIELDS = (*SCALAR_FIELDS, *SET_FIELDS)
MISMATCH_FIELDS = (
    "source_filename",
    "page",
    "field",
    "expected",
    "predicted",
    "flagged",
    "reasons",
)


@dataclass(frozen=True)
class FieldComparison:
    expected: Any
    predicted: Any
    correct: bool
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0


def normalize_id(value: Any) -> str:
    digits = "".join(char for char in str(value or "") if char.isascii() and char.isdigit())
    return digits.zfill(9) if digits and len(digits) <= 9 else digits


def normalize_name(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def normalize_date(value: Any) -> str:
    normalized, _ = normalize_received_date(str(value or ""))
    return normalized or ""


def normalize_entity_value(value: Any) -> str:
    return (normalize_entity(str(value)) or "").casefold()


def split_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        text = str(value).strip()
        if not text:
            return []
        values = text.split(";")
    return list(dict.fromkeys(
        digits
        for item in values
        if (digits := "".join(
            char for char in str(item) if char.isascii() and char.isdigit()
        ))
    ))


def normalize_field(field: str, value: Any) -> Any:
    if field in SET_FIELDS:
        return frozenset(split_values(value))
    if field == "id_number":
        return normalize_id(value)
    if field in {"first_name", "last_name"}:
        return normalize_name(value)
    if field == "received_date":
        return normalize_date(value)
    if field == "referring_entity":
        return normalize_entity_value(value)
    raise ValueError(f"Unknown evaluation field: {field}")


def compare_field(field: str, expected: Any, predicted: Any) -> FieldComparison:
    normalized_expected = normalize_field(field, expected)
    normalized_predicted = normalize_field(field, predicted)
    if field not in SET_FIELDS:
        return FieldComparison(
            normalized_expected,
            normalized_predicted,
            normalized_expected == normalized_predicted,
        )
    true_positive = len(normalized_expected & normalized_predicted)
    false_positive = len(normalized_predicted - normalized_expected)
    false_negative = len(normalized_expected - normalized_predicted)
    return FieldComparison(
        normalized_expected,
        normalized_predicted,
        normalized_expected == normalized_predicted,
        true_positive,
        false_positive,
        false_negative,
    )


def reason_mentions_field(field: str, reasons: Iterable[str]) -> bool:
    aliases = {
        "first_name": ("first name",),
        "last_name": ("last name",),
        "id_number": ("id number", "israeli id"),
        "central_numbers": ("central number",),
        "received_date": ("receipt date", "received date", "sample date"),
        "referring_entity": ("referring entity", "entity", "hmo", "hospital"),
        "authorization_number": ("authorization", "commitment", "requirement", "sample number"),
    }[field]
    normalized_reasons = [re.sub(r"\s+", " ", reason.casefold()) for reason in reasons]
    return any(alias in reason for alias in aliases for reason in normalized_reasons)


def is_flagged(
    field: str,
    predicted: Any,
    confidence: Any,
    reasons: Iterable[str],
) -> bool:
    empty = not normalize_field(field, predicted)
    try:
        low_confidence = float(confidence) < MIN_CONFIDENCE
    except (TypeError, ValueError):
        low_confidence = True
    return empty or low_confidence or reason_mentions_field(field, reasons)


def flag_calibration(rows: Iterable[dict[str, Any]], field: str) -> dict[str, Any]:
    total_wrong = flagged_wrong = silent_wrong = flagged_total = flagged_correct = 0
    silent_errors: list[dict[str, Any]] = []
    for row in rows:
        expected = row.get("expected", {}).get(field)
        predicted = row.get("predicted", {}).get(field)
        comparison = compare_field(field, expected, predicted)
        reasons = row.get("review_reasons", [])
        confidence = row.get("confidence", {}).get(field)
        flagged = is_flagged(field, predicted, confidence, reasons)
        if flagged:
            flagged_total += 1
            flagged_correct += comparison.correct
        if comparison.correct:
            continue
        total_wrong += 1
        if flagged:
            flagged_wrong += 1
        else:
            silent_wrong += 1
            silent_errors.append({
                "source_filename": row["source_filename"],
                "page": row["page"],
                "field": field,
                "expected": display_value(comparison.expected),
                "predicted": display_value(comparison.predicted),
                "flagged": False,
                "reasons": "; ".join(reasons),
            })
    return {
        "wrong": total_wrong,
        "flagged_wrong": flagged_wrong,
        "silent_wrong": silent_wrong,
        "error_recall": ratio(flagged_wrong, total_wrong),
        "flagged": flagged_total,
        "flagged_correct": flagged_correct,
        "flag_precision": ratio(flagged_wrong, flagged_total),
        "silent_errors": silent_errors,
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def display_value(value: Any) -> str:
    if isinstance(value, (set, frozenset)):
        return ";".join(sorted(value))
    return str(value or "")


def load_ground_truth(path: Path, split: str = "all") -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        lines = (line for line in source if not line.startswith("#"))
        reader = csv.DictReader(lines)
        if not reader.fieldnames or not set(GROUND_TRUTH_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"{path} is missing required ground-truth columns.")
        rows = []
        for row in reader:
            if not row.get("source_filename", "").strip():
                continue
            row_split = (row.get("split") or "").strip()
            if row_split not in {"dev", "test"}:
                raise ValueError(
                    f"Ground-truth row {row['source_filename']} page {row['page']} "
                    "must use split dev or test."
                )
            if split != "all" and row_split != split:
                continue
            row["source_filename"] = Path(row["source_filename"]).name
            try:
                row["page"] = int(row["page"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Ground-truth row {row['source_filename']} has an invalid page."
                ) from error
            if row["page"] < 1:
                raise ValueError(
                    f"Ground-truth row {row['source_filename']} must use a positive page number."
                )
            rows.append(row)
    return rows


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {
        "records": len(rows),
        "review_records": 0,
        "fields": {},
        "calibration": {},
        "silent_errors": [],
    }
    for row in rows:
        any_flagged = any(
            is_flagged(
                field,
                row.get("predicted", {}).get(field),
                row.get("confidence", {}).get(field),
                row.get("review_reasons", []),
            )
            for field in EVALUATED_FIELDS
        )
        results["review_records"] += any_flagged
    for field in EVALUATED_FIELDS:
        comparisons = [
            compare_field(
                field,
                row.get("expected", {}).get(field),
                row.get("predicted", {}).get(field),
            )
            for row in rows
        ]
        metric: dict[str, Any] = {
            "correct": sum(comparison.correct for comparison in comparisons),
            "total": len(comparisons),
            "accuracy": ratio(
                sum(comparison.correct for comparison in comparisons),
                len(comparisons),
            ),
        }
        if field in SET_FIELDS:
            true_positive = sum(item.true_positive for item in comparisons)
            false_positive = sum(item.false_positive for item in comparisons)
            false_negative = sum(item.false_negative for item in comparisons)
            metric.update({
                "precision": ratio(true_positive, true_positive + false_positive),
                "recall": ratio(true_positive, true_positive + false_negative),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
            })
        results["fields"][field] = metric
        calibration = flag_calibration(rows, field)
        results["calibration"][field] = {
            key: value for key, value in calibration.items() if key != "silent_errors"
        }
        results["silent_errors"].extend(calibration["silent_errors"])
    results["review_rate"] = ratio(results["review_records"], len(rows))
    results["id_check_digit"] = id_check_digit_rates(rows)
    results["silent_errors"].sort(
        key=lambda item: (
            item["flagged"],
            item["source_filename"].casefold(),
            item["page"],
            item["field"],
        )
    )
    return results


def id_check_digit_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from app.validation import normalize_israeli_id

    expected_ids = [
        normalize_id(row.get("expected", {}).get("id_number"))
        for row in rows
        if normalize_id(row.get("expected", {}).get("id_number"))
    ]
    predicted_ids = [
        normalize_id(row.get("predicted", {}).get("id_number"))
        for row in rows
        if normalize_id(row.get("predicted", {}).get("id_number"))
    ]
    expected_valid = sum(normalize_israeli_id(value)[1] for value in expected_ids)
    predicted_valid = sum(normalize_israeli_id(value)[1] for value in predicted_ids)
    return {
        "ground_truth_valid": expected_valid,
        "ground_truth_total": len(expected_ids),
        "ground_truth_rate": ratio(expected_valid, len(expected_ids)),
        "prediction_valid": predicted_valid,
        "prediction_total": len(predicted_ids),
        "prediction_rate": ratio(predicted_valid, len(predicted_ids)),
        "note": "Sample IDs are anonymized; a failing check digit is not proof of an OCR error.",
    }


def metrics_by_entity(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        entity = (
            normalize_entity_value(row.get("expected", {}).get("referring_entity"))
            or "(empty)"
        )
        grouped[entity].append(row)
    return {entity: aggregate_metrics(entity_rows) for entity, entity_rows in sorted(grouped.items())}
