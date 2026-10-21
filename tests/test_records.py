import tempfile
import unittest
from io import BytesIO
from pathlib import Path
import sqlite3

from PIL import Image

import app.main as main


class RecordPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.original_data_dir = main.DATA_DIR
        self.original_database_path = main.DATABASE_PATH
        main.DATA_DIR = Path(self.directory.name)
        main.DATABASE_PATH = main.DATA_DIR / "test.sqlite3"
        main.initialize_db()
        image = Image.new("RGB", (40, 60), "white")
        image_bytes = BytesIO()
        image.save(image_bytes, format="JPEG")
        self.page_image = image_bytes.getvalue()
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
            "preview_thumbnail": None,
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

        records = main.save_records("form.pdf", [(1, self.page_image), (2, self.page_image)], self.extracted)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["central_numbers"], ["C-10", "C-11"])
        self.assertEqual(records[1]["source_pages"], [1, 2])
        self.assertEqual(len(main.list_records()), 2)

    def test_preview_thumbnail_is_persisted_with_record(self) -> None:
        record = main.save_records(
            "form.pdf",
            [(1, self.page_image)],
            self.extracted,
        )[0]

        self.assertTrue(record["preview_thumbnail"])
        saved = main.list_records()[0]
        self.assertEqual(saved["preview_thumbnail"], record["preview_thumbnail"])

    def test_existing_database_gets_preview_column(self) -> None:
        old_database = Path(self.directory.name) / "old.sqlite3"
        connection = sqlite3.connect(old_database)
        try:
            connection.execute(
                """
                CREATE TABLE authorization_records (
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
        finally:
            connection.close()

        original_path = main.DATABASE_PATH
        main.DATABASE_PATH = old_database
        try:
            main.initialize_db()
            connection = sqlite3.connect(old_database)
            try:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(authorization_records)")
                }
            finally:
                connection.close()
            self.assertIn("preview_thumbnail", columns)
        finally:
            main.DATABASE_PATH = original_path

    def test_approval_is_cleared_after_later_correction(self) -> None:
        record = main.save_records("form.png", [(1, self.page_image)], self.extracted)[0]
        approved = main.update_record(record["id"], main.RecordCorrection(first_name="Maya", approved=True))
        corrected = main.update_record(record["id"], main.RecordCorrection(last_name="Levi"))

        self.assertTrue(approved["approved"])
        self.assertFalse(corrected["approved"])
        self.assertEqual(corrected["last_name"], "Levi")


if __name__ == "__main__":
    unittest.main()
