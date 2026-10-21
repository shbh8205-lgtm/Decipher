import asyncio
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import httpx
import pymupdf
from PIL import Image

import app.main as main


class ExtractionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.original_data_dir = main.DATA_DIR
        self.original_database_path = main.DATABASE_PATH
        main.DATA_DIR = Path(self.directory.name)
        main.DATABASE_PATH = main.DATA_DIR / "test.sqlite3"

    def tearDown(self) -> None:
        main.DATA_DIR = self.original_data_dir
        main.DATABASE_PATH = self.original_database_path
        self.directory.cleanup()

    def test_upload_extracts_and_returns_persisted_records(self) -> None:
        image = Image.new("RGB", (32, 32), "white")
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        extracted = {
            "first_name": "Maya",
            "last_name": "Cohen",
            "id_number": "123456782",
            "central_numbers": ["C-10"],
            "received_date": "2025-04-12",
            "referring_entity": "Clalit",
            "authorizations": [{
                "number": "A-100",
                "label": "commitment",
                "confidence": 0.92,
                "evidence": "Visible commitment",
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

        with patch.object(main, "extract_document", new=AsyncMock(return_value=extracted)):
            with TestClient(main.app) as client:
                response = client.post(
                    "/api/extract",
                    data={"external_processing_consent": "true"},
                    files=[("files", ("authorization.png", image_bytes.getvalue(), "image/png"))],
                )
                self.assertEqual(response.status_code, 200)
                record = response.json()["documents"][0]["records"][0]
                self.assertEqual(record["id_number"], "123456782")
                self.assertEqual(record["received_date"], "12/04/2025")
                self.assertEqual(record["authorization_number"], "A-100")
                thumbnail = response.json()["documents"][0]["preview_thumbnail"]
                self.assertTrue(thumbnail.startswith("/9j/"))
                self.assertEqual(client.get("/api/records").json()[0]["id"], record["id"])
                corrected = client.patch(
                    f"/api/records/{record['id']}",
                    json={"id_number": "123456789"},
                )
                self.assertIn(
                    "Invalid Israeli ID check digit or length",
                    corrected.json()["review_reasons"],
                )
                second_page = Image.new("RGB", (32, 32), "white")
                second_page_bytes = BytesIO()
                second_page.save(second_page_bytes, format="PNG")
                grouped = client.post(
                    "/api/extract",
                    data={"group_images": "true", "external_processing_consent": "true"},
                    files=[
                        ("files", ("page-one.png", image_bytes.getvalue(), "image/png")),
                        ("files", ("page-two.png", second_page_bytes.getvalue(), "image/png")),
                    ],
                )
                self.assertEqual(grouped.status_code, 200)
                grouped_document = grouped.json()["documents"][0]
                self.assertTrue(grouped_document["preview_thumbnail"].startswith("/9j/"))
                self.assertEqual(grouped_document["records"][0]["source_pages"], [1, 2])
                self.assertEqual(
                    grouped_document["records"][0]["source_filename"],
                    "page-one.png + page-two.png",
                )

    def test_gemini_request_keeps_key_in_header_and_sends_structured_image(self) -> None:
        extracted = {
            "status": "completed",
            "steps": [{
                "type": "model_output",
                "content": [{"type": "text", "text": json.dumps({"first_name": "Maya"})}],
            }],
        }
        requests = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=extracted)

        original_client = httpx.AsyncClient
        mock_transport = httpx.MockTransport(handler)

        with patch.object(main, "GEMINI_API_KEY", "test-key"), patch.object(
            main, "GEMINI_MODEL", "gemini-3.8-flash"
        ):
            with patch.object(
                main.httpx,
                "AsyncClient",
                side_effect=lambda **kwargs: original_client(transport=mock_transport, **kwargs),
            ):
                result = asyncio.run(main.extract_once([(1, b"jpeg-bytes")], "test prompt"))

        self.assertEqual(result, {"first_name": "Maya"})
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertNotIn("test-key", str(request.url))
        self.assertEqual(str(request.url), main.GEMINI_API_URL)
        self.assertEqual(request.headers["x-goog-api-key"], "test-key")
        body = json.loads(request.content)
        self.assertEqual(body["model"], "gemini-3.8-flash")
        self.assertEqual(body["input"][0], {"type": "text", "text": "test prompt"})
        self.assertEqual(body["input"][1]["type"], "image")
        self.assertEqual(body["input"][1]["mime_type"], "image/jpeg")
        self.assertEqual(body["response_format"]["mime_type"], "application/json")
        self.assertFalse(body["store"])

    def test_gemini_schema_maps_nullable_types(self) -> None:
        schema = main.gemini_response_schema(main.EXTRACTION_SCHEMA)
        self.assertEqual(schema["properties"]["id_number"]["type"], "STRING")
        self.assertTrue(schema["properties"]["id_number"]["nullable"])
        self.assertEqual(schema["properties"]["authorizations"]["type"], "ARRAY")

    def test_gemini_404_identifies_model_and_network_filter_possibility(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={"error": {"message": "models/example not found"}},
            )

        original_client = httpx.AsyncClient
        mock_transport = httpx.MockTransport(handler)
        with patch.object(main, "GEMINI_API_KEY", "test-key"):
            with patch.object(
                main.httpx,
                "AsyncClient",
                side_effect=lambda **kwargs: original_client(transport=mock_transport, **kwargs),
            ):
                with self.assertRaises(main.HTTPException) as context:
                    asyncio.run(main.extract_once([(1, b"jpeg-bytes")], "test prompt"))

        self.assertEqual(context.exception.status_code, 502)
        self.assertIn(main.GEMINI_MODEL, context.exception.detail)
        self.assertIn("NetFree", context.exception.detail)
        self.assertIn("models/example not found", context.exception.detail)

    def test_gemini_interaction_without_model_output_is_an_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"status": "completed", "steps": [{"type": "user_input", "content": []}]},
            )

        original_client = httpx.AsyncClient
        mock_transport = httpx.MockTransport(handler)
        with patch.object(main, "GEMINI_API_KEY", "test-key"):
            with patch.object(
                main.httpx,
                "AsyncClient",
                side_effect=lambda **kwargs: original_client(transport=mock_transport, **kwargs),
            ):
                with self.assertRaises(main.HTTPException) as context:
                    asyncio.run(main.extract_once([(1, b"jpeg-bytes")], "test prompt"))

        self.assertEqual(context.exception.status_code, 502)
        self.assertIn("no text", context.exception.detail)

    def test_upload_requires_external_processing_consent(self) -> None:
        with TestClient(main.app) as client:
            response = client.post(
                "/api/extract",
                files=[("files", ("authorization.png", b"not read", "image/png"))],
            )
            self.assertEqual(response.status_code, 400)

    def test_pdf_over_page_limit_reports_page_count(self) -> None:
        document = pymupdf.open()
        for _ in range(main.MAX_PAGES + 1):
            document.new_page()
        pdf_bytes = document.tobytes()
        document.close()

        with self.assertRaisesRegex(
            ValueError,
            f"PDF has {main.MAX_PAGES + 1} pages; the limit is {main.MAX_PAGES}.",
        ):
            main.normalize_document("many-pages.pdf", pdf_bytes)


if __name__ == "__main__":
    unittest.main()
