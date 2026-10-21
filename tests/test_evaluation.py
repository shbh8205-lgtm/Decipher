import unittest

from scripts.evaluation_utils import (
    compare_field,
    flag_calibration,
    is_flagged,
    normalize_field,
)


class EvaluationNormalizationTests(unittest.TestCase):
    def test_id_normalization_keeps_leading_zeroes(self) -> None:
        self.assertEqual(normalize_field("id_number", " 123-456-78 "), "012345678")

    def test_name_and_date_normalization(self) -> None:
        self.assertEqual(normalize_field("first_name", "  Maya\t Cohen "), "maya cohen")
        self.assertEqual(normalize_field("received_date", "2024-03-07"), "07/03/2024")

    def test_entity_aliases_normalize_before_comparison(self) -> None:
        comparison = compare_field("referring_entity", "כללית", "Clalit")
        self.assertTrue(comparison.correct)

    def test_number_sets_ignore_separators_and_order(self) -> None:
        comparison = compare_field(
            "central_numbers",
            "937309; 937310",
            ["937-310", "937309"],
        )
        self.assertTrue(comparison.correct)
        self.assertEqual(comparison.true_positive, 2)
        self.assertEqual(comparison.false_positive, 0)
        self.assertEqual(comparison.false_negative, 0)

    def test_empty_prediction_and_field_reason_are_flags(self) -> None:
        self.assertTrue(is_flagged("first_name", "", 0.99, []))
        self.assertTrue(is_flagged("central_numbers", ["123456"], 0.99, ["Low confidence: central numbers"]))
        self.assertFalse(is_flagged("id_number", "123456782", 0.9, []))


class FlagCalibrationTests(unittest.TestCase):
    def test_reports_error_recall_flag_precision_and_silent_error_rows(self) -> None:
        rows = [
            {
                "source_filename": "a.pdf",
                "page": 1,
                "expected": {"first_name": "Maya"},
                "predicted": {"first_name": "Mia"},
                "confidence": {"first_name": 0.95},
                "review_reasons": [],
            },
            {
                "source_filename": "b.pdf",
                "page": 2,
                "expected": {"first_name": "Leah"},
                "predicted": {"first_name": "Lia"},
                "confidence": {"first_name": 0.2},
                "review_reasons": [],
            },
            {
                "source_filename": "c.pdf",
                "page": 3,
                "expected": {"first_name": "Noa"},
                "predicted": {"first_name": "Noa"},
                "confidence": {"first_name": 0.3},
                "review_reasons": [],
            },
        ]

        result = flag_calibration(rows, "first_name")

        self.assertEqual(result["wrong"], 2)
        self.assertEqual(result["flagged_wrong"], 1)
        self.assertEqual(result["silent_wrong"], 1)
        self.assertEqual(result["error_recall"], 0.5)
        self.assertEqual(result["flagged"], 2)
        self.assertEqual(result["flag_precision"], 0.5)
        self.assertEqual(result["silent_errors"][0]["source_filename"], "a.pdf")
        self.assertFalse(result["silent_errors"][0]["flagged"])


if __name__ == "__main__":
    unittest.main()
