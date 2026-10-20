import unittest
from datetime import date, timedelta

from app.validation import normalize_israeli_id, normalize_received_date


class IsraeliIdTests(unittest.TestCase):
    def test_valid_id_is_padded_and_preserves_leading_zeroes(self) -> None:
        self.assertEqual(normalize_israeli_id("123456782"), ("123456782", True))
        self.assertEqual(normalize_israeli_id("12345674"), ("012345674", True))

    def test_invalid_id_is_returned_padded_and_flagged(self) -> None:
        self.assertEqual(normalize_israeli_id("123456789"), ("123456789", False))

    def test_missing_or_overlong_id_is_invalid(self) -> None:
        self.assertEqual(normalize_israeli_id(None), (None, False))
        self.assertEqual(normalize_israeli_id("1234567890"), ("1234567890", False))


class ReceivedDateTests(unittest.TestCase):
    def test_supported_formats_are_normalized(self) -> None:
        self.assertEqual(normalize_received_date("2024-03-07"), ("07/03/2024", None))
        self.assertEqual(normalize_received_date("7.3.2024"), ("07/03/2024", None))

    def test_future_old_and_unreadable_dates_are_flagged(self) -> None:
        future = (date.today() + timedelta(days=1)).strftime("%d/%m/%Y")
        self.assertIn("future", normalize_received_date(future)[1])
        self.assertIn("old", normalize_received_date("31/12/1999")[1])
        self.assertEqual(normalize_received_date("not a date"), (None, "Unrecognized sample receipt date"))

    def test_missing_date_is_flagged(self) -> None:
        self.assertEqual(normalize_received_date(None), (None, "Missing sample receipt date"))


if __name__ == "__main__":
    unittest.main()
