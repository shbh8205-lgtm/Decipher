import argparse
import asyncio
import csv
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import main as app
from app.barcodes import decode_page_barcodes
from scripts.evaluation_pipeline import (
    cache_path,
    extract_raw_pair,
    load_page_image,
    prediction_records,
    read_cache,
    write_cache,
)
from scripts.evaluation_utils import (
    EVALUATED_FIELDS,
    MISMATCH_FIELDS,
    SCALAR_FIELDS,
    SET_FIELDS,
    aggregate_metrics,
    compare_field,
    display_value,
    is_flagged,
    load_ground_truth,
    metrics_by_entity,
)


FIELD_LABELS = {
    "first_name": "First name",
    "last_name": "Last name",
    "id_number": "ID number",
    "central_numbers": "Central numbers",
    "received_date": "Received date",
    "referring_entity": "Referring entity",
    "authorization_number": "Authorization number",
}


def empty_prediction(reason: str = "Missing predicted record") -> dict[str, Any]:
    return {
        **{field: None for field in SCALAR_FIELDS},
        **{field: [] for field in SET_FIELDS},
        "confidence": {},
        "review_reasons": [reason],
    }


def expected_values(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **{field: row.get(field, "") for field in SCALAR_FIELDS},
        "central_numbers": row.get("central_numbers", ""),
        "authorization_number": row.get("authorization_number", ""),
    }


def pair_predictions(
    truth_rows: list[dict[str, Any]],
    prediction_map: dict[tuple[str, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in truth_rows:
        grouped[(row["source_filename"].casefold(), row["page"])].append(row)

    combined = []
    for key, expected_rows in grouped.items():
        candidates = list(prediction_map.get(key, []))
        assigned: dict[int, dict[str, Any]] = {}
        unused = set(range(len(candidates)))
        for truth_index, truth in enumerate(expected_rows):
            expected_auth = compare_field(
                "authorization_number",
                truth.get("authorization_number", ""),
                "",
            ).expected
            match_index = next((
                index for index in sorted(unused)
                if compare_field(
                    "authorization_number",
                    truth.get("authorization_number", ""),
                    candidates[index].get("authorization_number"),
                ).expected == compare_field(
                    "authorization_number",
                    truth.get("authorization_number", ""),
                    candidates[index].get("authorization_number"),
                ).predicted
                and expected_auth
            ), None)
            if match_index is None and unused:
                match_index = min(unused)
            if match_index is not None:
                assigned[truth_index] = candidates[match_index]
                unused.remove(match_index)
            else:
                assigned[truth_index] = empty_prediction()
        for index, truth in enumerate(expected_rows):
            combined.append({
                "source_filename": truth["source_filename"],
                "page": truth["page"],
                "entity": truth.get("referring_entity", ""),
                "expected": expected_values(truth),
                "predicted": {
                    field: assigned[index].get(field)
                    for field in EVALUATED_FIELDS
                },
                "confidence": assigned[index].get("confidence", {}),
                "review_reasons": assigned[index].get("review_reasons", []),
            })
        for index in sorted(unused):
            prediction = candidates[index]
            expected = expected_values(expected_rows[0])
            expected["authorization_number"] = ""
            combined.append({
                "source_filename": expected_rows[0]["source_filename"],
                "page": expected_rows[0]["page"],
                "entity": expected_rows[0].get("referring_entity", ""),
                "expected": expected,
                "predicted": {
                    field: prediction.get(field)
                    for field in EVALUATED_FIELDS
                },
                "confidence": prediction.get("confidence", {}),
                "review_reasons": prediction.get("review_reasons", []),
            })
    return combined


def read_db_predictions(database_path: Path) -> tuple[dict[tuple[str, int], list[dict[str, Any]]], int]:
    if not database_path.is_file():
        raise FileNotFoundError(f"Database not found: {database_path}")
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(authorization_records)")
        }
        required = {
            "source_filename", "source_pages", "first_name", "last_name", "id_number",
            "central_numbers", "received_date", "referring_entity", "authorization_number",
            "field_confidence", "review_reasons",
        }
        if not required.issubset(columns):
            raise ValueError("Database authorization_records table is missing evaluation fields.")
        rows = connection.execute(
            "SELECT source_filename, source_pages, first_name, last_name, id_number, "
            "central_numbers, received_date, referring_entity, authorization_number, "
            "field_confidence, review_reasons FROM authorization_records"
        ).fetchall()
    finally:
        connection.close()

    predictions: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        confidence = json.loads(row["field_confidence"] or "{}")
        reasons = json.loads(row["review_reasons"] or "[]")
        centrals = json.loads(row["central_numbers"] or "[]")
        record = {
            "first_name": row["first_name"],
            "last_name": row["last_name"],
            "id_number": row["id_number"],
            "central_numbers": centrals if isinstance(centrals, list) else [centrals],
            "received_date": row["received_date"],
            "referring_entity": row["referring_entity"],
            "authorization_number": row["authorization_number"],
            "confidence": confidence,
            "review_reasons": reasons if isinstance(reasons, list) else [str(reasons)],
        }
        for page in json.loads(row["source_pages"] or "[]"):
            predictions[(Path(row["source_filename"]).name.casefold(), int(page))].append(record)
    return predictions, len(rows)


async def live_prediction_map(
    truth_rows: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped = {
        (row["source_filename"], row["page"]): row
        for row in truth_rows
    }
    predictions: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for (filename, page), truth in grouped.items():
        image_bytes = load_page_image(filename, page, truth.get("page_image"))
        first, second = await extract_raw_pair(image_bytes)
        barcodes = await asyncio.to_thread(decode_page_barcodes, image_bytes)
        write_cache(
            cache_path(filename, page),
            filename,
            page,
            image_bytes,
            first,
            second,
            barcodes,
        )
        predictions[(filename.casefold(), page)] = prediction_records(first, second, barcodes)
    return predictions


def cached_prediction_map(
    truth_rows: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped = {
        (row["source_filename"], row["page"]): row
        for row in truth_rows
    }
    predictions = {}
    for filename, page in sorted(grouped):
        cached = read_cache(cache_path(filename, page), filename, page)
        image_bytes = load_page_image(filename, page, grouped[(filename, page)].get("page_image"))
        image_digest = hashlib.sha256(image_bytes).hexdigest()
        if cached.get("image_sha256") != image_digest:
            raise ValueError(
                f"Cached extraction for {filename} page {page} does not match its rendered image. "
                "Regenerate the draft/cache before evaluating."
            )
        predictions[(filename.casefold(), page)] = prediction_records(
            cached["first"],
            cached["second"],
            cached.get("decoded_central_barcodes", []),
        )
    return predictions


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def metrics_table(metrics: dict[str, Any]) -> list[str]:
    lines = [
        "| Field | Exact match | Correct / rows | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for field in EVALUATED_FIELDS:
        item = metrics["fields"][field]
        precision = percent(item.get("precision")) if field in SET_FIELDS else "n/a"
        recall = percent(item.get("recall")) if field in SET_FIELDS else "n/a"
        lines.append(
            f"| {FIELD_LABELS[field]} | {percent(item['accuracy'])} | "
            f"{item['correct']} / {item['total']} | {precision} | {recall} |"
        )
    return lines


def calibration_table(metrics: dict[str, Any]) -> list[str]:
    lines = [
        "| Field | Wrong | Error recall flagged | Silent errors | Flag precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for field in EVALUATED_FIELDS:
        item = metrics["calibration"][field]
        lines.append(
            f"| {FIELD_LABELS[field]} | {item['wrong']} | "
            f"{percent(item['error_recall'])} | {item['silent_wrong']} | "
            f"{percent(item['flag_precision'])} |"
        )
    return lines


def build_report(
    metrics: dict[str, Any],
    by_entity: dict[str, dict[str, Any]],
    split: str,
    mode: str,
    prediction_count: int,
) -> str:
    lines = [
        "# Extraction evaluation report",
        "",
        f"- Split: **{split}**",
        f"- Prediction source: **{mode}**",
        f"- Labeled authorization rows: **{metrics['records']}**",
        f"- Available saved/predicted records: **{prediction_count}**",
        "",
    ]
    if not metrics["records"]:
        lines.extend([
            "**No labeled rows exist for this split. Accuracy and calibration are not "
            "calculable; this is not a zero-accuracy result. Add manually verified rows to "
            "`eval/ground_truth.csv` and rerun evaluation.**",
            "",
        ])
    lines.extend([
        "## Overall exact-match accuracy",
        "",
        *metrics_table(metrics),
        "",
        "For set fields, precision/recall count individual normalized numbers; exact match "
        "requires the whole predicted set to equal the labeled set.",
        "",
        "## Flag calibration",
        "",
        "A field is flagged when prediction confidence is below the app threshold, the "
        "prediction is empty, or a review reason mentions that field.",
        "",
        *calibration_table(metrics),
        "",
    ])
    lines.extend([
        "",
        f"Records with at least one flagged field (review workload): "
        f"**{metrics['review_records']} / {metrics['records']} "
        f"({percent(metrics['review_rate'])})**.",
        "",
        "## Israeli ID check-digit rates",
        "",
    ])
    id_metrics = metrics["id_check_digit"]
    lines.extend([
        f"- Ground truth: {id_metrics['ground_truth_valid']} / "
        f"{id_metrics['ground_truth_total']} ({percent(id_metrics['ground_truth_rate'])})",
        f"- Predictions: {id_metrics['prediction_valid']} / "
        f"{id_metrics['prediction_total']} ({percent(id_metrics['prediction_rate'])})",
        f"- Note: {id_metrics['note']}",
        "",
        "## Accuracy by referring entity (ground-truth entity)",
        "",
    ])
    for entity, entity_metrics in by_entity.items():
        lines.extend([
            f"### {entity}",
            "",
            f"Rows: **{entity_metrics['records']}**; records needing review: "
            f"**{entity_metrics['review_records']} / {entity_metrics['records']} "
            f"({percent(entity_metrics['review_rate'])})**.",
            "",
            *metrics_table(entity_metrics),
            "",
            "Flag calibration:",
            "",
            *calibration_table(entity_metrics),
            "",
            f"ID check digits: ground truth "
            f"{entity_metrics['id_check_digit']['ground_truth_valid']} / "
            f"{entity_metrics['id_check_digit']['ground_truth_total']} "
            f"({percent(entity_metrics['id_check_digit']['ground_truth_rate'])}); "
            f"predictions {entity_metrics['id_check_digit']['prediction_valid']} / "
            f"{entity_metrics['id_check_digit']['prediction_total']} "
            f"({percent(entity_metrics['id_check_digit']['prediction_rate'])}).",
            "",
        ])
    lines.extend([
        "## Silent errors",
        "",
        "Silent errors are wrong values without a field-level review flag. See "
        "`eval/mismatches.csv` for all mismatches, with silent errors sorted first.",
        "",
    ])
    if metrics["silent_errors"]:
        lines.extend([
            "| Source | Page | Field | Expected | Predicted |",
            "|---|---:|---|---|---|",
        ])
        for error in metrics["silent_errors"]:
            lines.append(
                f"| {error['source_filename']} | {error['page']} | "
                f"{FIELD_LABELS[error['field']]} | {error['expected']} | {error['predicted']} |"
            )
    else:
        lines.append("None recorded." if metrics["records"] else "Not available without labeled rows.")
    lines.extend(["", ""])
    return "\n".join(lines)


def write_outputs(
    rows: list[dict[str, Any]],
    split: str,
    mode: str,
    prediction_count: int,
) -> dict[str, Any]:
    report_path = ROOT / "eval" / "report.md"
    mismatch_path = ROOT / "eval" / "mismatches.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = aggregate_metrics(rows)
    entity_metrics = metrics_by_entity(rows)
    mismatch_rows = []
    for row in rows:
        for field in EVALUATED_FIELDS:
            comparison = compare_field(
                field,
                row["expected"].get(field),
                row["predicted"].get(field),
            )
            if comparison.correct:
                continue
            reasons = row.get("review_reasons", [])
            flagged = is_flagged(
                field,
                row["predicted"].get(field),
                row.get("confidence", {}).get(field),
                reasons,
            )
            mismatch_rows.append({
                "source_filename": row["source_filename"],
                "page": row["page"],
                "field": field,
                "expected": display_value(comparison.expected),
                "predicted": display_value(comparison.predicted),
                "flagged": "yes" if flagged else "no",
                "reasons": "; ".join(reasons),
            })
    mismatch_rows.sort(key=lambda item: (
        item["flagged"] == "yes",
        item["source_filename"].casefold(),
        item["page"],
        item["field"],
    ))
    with mismatch_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=MISMATCH_FIELDS, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(mismatch_rows)
    report_path.write_text(
        build_report(metrics, entity_metrics, split, mode, prediction_count),
        encoding="utf-8",
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure extraction accuracy against labeled rows.")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--live", action="store_true")
    mode_group.add_argument("--cache", action="store_true")
    mode_group.add_argument("--from-db", action="store_true")
    args = parser.parse_args()
    mode = "live" if args.live else "cache" if args.cache else "from-db"
    try:
        truth_rows = load_ground_truth(ROOT / "eval" / "ground_truth.csv", args.split)
        if args.from_db:
            prediction_map, count = read_db_predictions(app.DATABASE_PATH)
        elif args.cache:
            prediction_map = cached_prediction_map(truth_rows)
            count = sum(len(items) for items in prediction_map.values())
        else:
            prediction_map = asyncio.run(live_prediction_map(truth_rows))
            count = sum(len(items) for items in prediction_map.values())
        paired = pair_predictions(truth_rows, prediction_map)
        metrics = write_outputs(paired, args.split, mode, count)
        report_path = ROOT / "eval" / "report.md"
        print(report_path.read_text(encoding="utf-8"))
        if not metrics["records"]:
            print("No labeled rows were available; no accuracy numbers were computed.")
    except Exception as error:
        message = f"Evaluation failed: {type(error).__name__}: {error}"
        print(message, file=sys.stderr)
        output_path = ROOT / "eval" / "report.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"# Extraction evaluation report\n\n{message}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
