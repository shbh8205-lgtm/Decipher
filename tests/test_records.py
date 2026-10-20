import tempfile
import unittest
from pathlib import Path

import app.main as main


class RecordPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.original_data_dir = main.DATA_DIR
        self.original_database_path = main.DATABASE_PATH
        main.DATA_DIR = Path(self.directory.name)
        main.DATABASE_PATH = main.DATA_DIR / "test.sqlite3"
        main.initialize_db()
        self.extracted = {
            "first_name": "Maya",
            "last_name": "Cohen",
            "id_number": "123456782",
            "central_numbers": ["C-10", "C-11"],
            "received_date": "12/04/2025",
            "referring_entity": "Clalit",
            "authorizations": [{
                "number": "A-100",
                "label": "commitment",
                "confidence": 0.92,
                "evidence": "Visible commitment number",
            }],
            "field_confidence": {
                "first_name": 0.92,
                "last_name": 0.92,
                "id_number": 0.92,
                "central_numbers": 0.92,
                "received_date": 0.92,
                "referring_entity": 0.92,
            },
            "evidence": {
                "first_name": "Maya",
                "last_name": "Cohen",
                "id_number": "123456782",
                "central_numbers": "Central sticker",
                "received_date": "Received stamp",
                "referring_entity": "Clalit logo",
            },
            "notes": [],
        }

    def tearDown(self) -> None:
        main.DATA_DIR = self.original_data_dir
        main.DATABASE_PATH = self.original_database_path
        self.directory.cleanup()

    def test_authorizations_are_persisted_as_separate_records(self) -> None:
        self.extracted["authorizations"].append({
            "number": "A-101",
            "label": "requirement",
            "confidence": 0.88,
            "evidence": "Visible requirement number",
        })

        records = main.save_records("form.pdf", [(1, b"page"), (2, b"page")], self.extracted)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["central_numbers"], ["C-10", "C-11"])
        self.assertEqual(records[1]["source_pages"], [1, 2])
        self.assertEqual(len(main.list_records()), 2)

    def test_approval_is_cleared_after_later_correction(self) -> None:
        record = main.save_records("form.png", [(1, b"page")], self.extracted)[0]
        approved = main.update_record(record["id"], main.RecordCorrection(first_name="Maya", approved=True))
        corrected = main.update_record(record["id"], main.RecordCorrection(last_name="Levi"))

        self.assertTrue(approved["approved"])
        self.assertFalse(corrected["approved"])
        self.assertEqual(corrected["last_name"], "Levi")


if __name__ == "__main__":
    unittest.main()
