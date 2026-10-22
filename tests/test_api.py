import asyncio
import base64
import csv
import json
import os
import tempfile
import threading
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import bcrypt
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
        self.original_app_users = os.environ.get("APP_USERS")
        self.original_app_password_hash = os.environ.get("APP_PASSWORD_HASH")
        os.environ["APP_USERS"] = (
            "test:" + bcrypt.hashpw(b"test-password", bcrypt.gensalt()).decode("ascii")
        )
        os.environ.pop("APP_PASSWORD_HASH", None)
        main._RATE_BUCKETS.clear()

    def tearDown(self) -> None:
        main.DATA_DIR = self.original_data_dir
        main.DATABASE_PATH = self.original_database_path
        if self.original_app_users is None:
            os.environ.pop("APP_USERS", None)
        else:
            os.environ["APP_USERS"] = self.original_app_users
        if self.original_app_password_hash is None:
            os.environ.pop("APP_PASSWORD_HASH", None)
        else:
            os.environ["APP_PASSWORD_HASH"] = self.original_app_password_hash
        main._RATE_BUCKETS.clear()
        self.directory.cleanup()

    def authenticated_client(self) -> TestClient:
        client = TestClient(main.app, base_url="https://testserver")
        response = client.post(
            "/login",
            data={"username": "test", "password": "test-password"},
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        return client

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
            with self.authenticated_client() as client:
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
                exported = client.get("/api/records.csv")
                self.assertEqual(exported.status_code, 200)
                self.assertIn("attachment;", exported.headers["content-disposition"])
                rows = list(csv.DictReader(exported.text.lstrip("\ufeff").splitlines()))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["ID Number (text)"], '="123456782"')
                self.assertEqual(rows[0]["Source Filename"], "authorization.png")
                self.assertEqual(rows[0]["Source Page(s)"], "1")
                self.assertEqual(rows[0]["Authorization Number"], "A-100")
                self.assertEqual(rows[0]["Central Verification"], "C-10: ocr_only")
                corrected = client.patch(
                    f"/api/records/{record['id']}",
                    json={"id_number": "123456789"},
                )
                self.assertIn(
                    "Invalid Israeli ID check digit or length",
                    corrected.json()["review_reasons"],
                )
                central_correction = client.patch(
                    f"/api/records/{record['id']}",
                    json={"central_numbers": ["C-11"]},
                )
                self.assertEqual(
                    central_correction.json()["barcode_verification"][0]["status"],
                    "manual",
                )
                deleted = client.delete(f"/api/records/{record['id']}")
                self.assertEqual(deleted.status_code, 204)
                self.assertEqual(client.get("/api/records").json(), [])
                missing = client.delete(f"/api/records/{record['id']}")
                self.assertEqual(missing.status_code, 404)
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

    def test_anthropic_request_sends_key_image_and_json_schema(self) -> None:
        extracted = {
            "content": [{"type": "text", "text": json.dumps({"first_name": "Maya"})}],
        }
        requests = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=extracted)

        original_client = httpx.AsyncClient
        mock_transport = httpx.MockTransport(handler)
        with patch.object(main, "ANTHROPIC_API_KEY", "anthropic-key"), patch.object(
            main, "ANTHROPIC_MODEL", "claude-test-model"
        ), patch.object(main, "GEMINI_API_KEY", ""), patch.object(
            main, "GEMINI_API_KEY_BACKUP", ""
        ), patch.object(main, "OPENROUTER_API_KEY", ""):
            with patch.object(
                main.httpx,
                "AsyncClient",
                side_effect=lambda **kwargs: original_client(transport=mock_transport, **kwargs),
            ):
                result = asyncio.run(main.extract_once([(1, b"jpeg-bytes")], "test prompt"))

        self.assertEqual(result, {"first_name": "Maya"})
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(str(request.url), main.ANTHROPIC_API_URL)
        self.assertEqual(request.headers["x-api-key"], "anthropic-key")
        self.assertEqual(request.headers["anthropic-version"], "2023-06-01")
        body = json.loads(request.content)
        self.assertEqual(body["model"], "claude-test-model")
        self.assertEqual(body["max_tokens"], 4096)
        self.assertEqual(body["messages"][0]["content"][0], {"type": "text", "text": "test prompt"})
        self.assertEqual(
            body["messages"][0]["content"][1],
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(b"jpeg-bytes").decode("ascii"),
                },
            },
        )
        self.assertEqual(
            body["output_config"]["format"],
            {"type": "json_schema", "schema": main.EXTRACTION_SCHEMA},
        )

    def test_anthropic_falls_back_to_gemini_after_overload_retries(self) -> None:
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
            if request.url.host == "api.anthropic.com":
                return httpx.Response(529)
            return httpx.Response(200, json=extracted)

        original_client = httpx.AsyncClient
        mock_transport = httpx.MockTransport(handler)
        with patch.object(main, "ANTHROPIC_API_KEY", "anthropic-key"), patch.object(
            main, "GEMINI_API_KEY", "gemini-key"
        ), patch.object(main, "GEMINI_API_KEY_BACKUP", ""), patch.object(
            main, "OPENROUTER_API_KEY", ""
        ), patch.object(main.asyncio, "sleep", new=AsyncMock()):
            with patch.object(
                main.httpx,
                "AsyncClient",
                side_effect=lambda **kwargs: original_client(transport=mock_transport, **kwargs),
            ):
                result = asyncio.run(main.extract_once([(1, b"jpeg-bytes")], "test prompt"))

        self.assertEqual(result, {"first_name": "Maya"})
        self.assertEqual(len(requests), 4)
        self.assertEqual(str(requests[0].url), main.ANTHROPIC_API_URL)
        self.assertEqual(requests[0].headers["x-api-key"], "anthropic-key")
        self.assertEqual(str(requests[2].url), main.ANTHROPIC_API_URL)
        self.assertEqual(str(requests[3].url), main.GEMINI_API_URL)
        self.assertEqual(requests[3].headers["x-goog-api-key"], "gemini-key")

    def test_anthropic_400_and_418_fall_back_to_gemini(self) -> None:
        extracted = {
            "status": "completed",
            "steps": [{
                "type": "model_output",
                "content": [{"type": "text", "text": json.dumps({"first_name": "Maya"})}],
            }],
        }

        for status_code in (400, 418):
            requests = []

            async def handler(request: httpx.Request) -> httpx.Response:
                requests.append(request)
                if request.url.host == "api.anthropic.com":
                    return httpx.Response(status_code)
                return httpx.Response(200, json=extracted)

            original_client = httpx.AsyncClient
            mock_transport = httpx.MockTransport(handler)
            with self.subTest(status_code=status_code):
                with patch.object(main, "ANTHROPIC_API_KEY", "anthropic-key"), patch.object(
                    main, "GEMINI_API_KEY", "gemini-key"
                ), patch.object(main, "GEMINI_API_KEY_BACKUP", ""), patch.object(
                    main, "OPENROUTER_API_KEY", ""
                ):
                    with patch.object(
                        main.httpx,
                        "AsyncClient",
                        side_effect=lambda **kwargs: original_client(
                            transport=mock_transport, **kwargs
                        ),
                    ):
                        result = asyncio.run(
                            main.extract_once([(1, b"jpeg-bytes")], "test prompt")
                        )

            self.assertEqual(result, {"first_name": "Maya"})
            self.assertEqual(len(requests), 2)
            self.assertEqual(str(requests[0].url), main.ANTHROPIC_API_URL)
            self.assertEqual(str(requests[1].url), main.GEMINI_API_URL)

    def test_status_lists_anthropic_before_fallback_providers(self) -> None:
        with patch.object(main, "ANTHROPIC_API_KEY", "anthropic-key"), patch.object(
            main, "GEMINI_API_KEY", "gemini-key"
        ), patch.object(main, "GEMINI_API_KEY_BACKUP", "backup-key"), patch.object(
            main, "OPENROUTER_API_KEY", "openrouter-key"
        ):
            result = main.status()

        self.assertTrue(result["configured"])
        self.assertEqual(result["model"], main.ANTHROPIC_MODEL)
        self.assertEqual(
            result["provider"],
            "Anthropic primary (claude-sonnet-5-5) + "
            f"Gemini fallback ({main.GEMINI_MODEL}) + "
            f"Gemini fallback ({main.GEMINI_MODEL}) + "
            f"OpenRouter fallback ({main.OPENROUTER_MODEL})",
        )

    def test_gemini_falls_back_to_openrouter_after_primary_quota_error(self) -> None:
        extracted = {
            "choices": [{
                "message": {"content": json.dumps({"first_name": "Maya"})},
            }],
        }
        requests = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.host == "generativelanguage.googleapis.com":
                return httpx.Response(429)
            return httpx.Response(200, json=extracted)

        original_client = httpx.AsyncClient
        mock_transport = httpx.MockTransport(handler)
        with patch.object(main, "GEMINI_API_KEY", "primary-key"), patch.object(
            main, "GEMINI_API_KEY_BACKUP", ""
        ), patch.object(
            main, "OPENROUTER_API_KEY", "openrouter-key"
        ):
            with patch.object(main, "OPENROUTER_MODEL", "openai/gpt-4o-mini"):
                with patch.object(
                    main.httpx,
                    "AsyncClient",
                    side_effect=lambda **kwargs: original_client(transport=mock_transport, **kwargs),
                ):
                    result = asyncio.run(main.extract_once([(1, b"jpeg-bytes")], "test prompt"))

        self.assertEqual(result, {"first_name": "Maya"})
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].headers["x-goog-api-key"], "primary-key")
        self.assertEqual(str(requests[1].url), main.OPENROUTER_API_URL)
        self.assertEqual(requests[1].headers["Authorization"], "Bearer openrouter-key")
        body = json.loads(requests[1].content)
        self.assertEqual(body["model"], "openai/gpt-4o-mini")
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(body["response_format"]["json_schema"]["schema"], main.EXTRACTION_SCHEMA)
        self.assertEqual(body["messages"][0]["content"][0], {"type": "text", "text": "test prompt"})
        self.assertEqual(
            body["messages"][0]["content"][1]["image_url"]["url"],
            "data:image/jpeg;base64," + base64.b64encode(b"jpeg-bytes").decode("ascii"),
        )

    def test_gemini_backup_is_tried_before_openrouter(self) -> None:
        extracted = {
            "choices": [{
                "message": {"content": json.dumps({"first_name": "Maya"})},
            }],
        }
        requests = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.host == "generativelanguage.googleapis.com":
                return httpx.Response(429)
            return httpx.Response(200, json=extracted)

        original_client = httpx.AsyncClient
        mock_transport = httpx.MockTransport(handler)
        with patch.object(main, "GEMINI_API_KEY", "primary-key"), patch.object(
            main, "GEMINI_API_KEY_BACKUP", "backup-key"
        ), patch.object(
            main, "OPENROUTER_API_KEY", "openrouter-key"
        ):
            with patch.object(
                main.httpx,
                "AsyncClient",
                side_effect=lambda **kwargs: original_client(transport=mock_transport, **kwargs),
            ):
                result = asyncio.run(main.extract_once([(1, b"jpeg-bytes")], "test prompt"))

        self.assertEqual(result, {"first_name": "Maya"})
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[0].headers["x-goog-api-key"], "primary-key")
        self.assertEqual(requests[1].headers["x-goog-api-key"], "backup-key")
        self.assertEqual(str(requests[2].url), main.OPENROUTER_API_URL)

    def test_gemini_does_not_fall_back_for_model_not_found(self) -> None:
        requests = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(404, json={"error": {"message": "model not found"}})

        original_client = httpx.AsyncClient
        mock_transport = httpx.MockTransport(handler)
        with patch.object(main, "GEMINI_API_KEY", "primary-key"), patch.object(
            main, "GEMINI_API_KEY_BACKUP", "backup-key"
        ), patch.object(
            main, "OPENROUTER_API_KEY", "openrouter-key"
        ):
            with patch.object(
                main.httpx,
                "AsyncClient",
                side_effect=lambda **kwargs: original_client(transport=mock_transport, **kwargs),
            ):
                with self.assertRaises(main.HTTPException):
                    asyncio.run(main.extract_once([(1, b"jpeg-bytes")], "test prompt"))

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].headers["x-goog-api-key"], "primary-key")

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

    def test_gemini_503_is_retried_and_succeeds(self) -> None:
        response = {
            "status": "completed",
            "steps": [{
                "type": "model_output",
                "content": [{"type": "text", "text": json.dumps({"first_name": "Maya"})}],
            }],
        }
        attempts = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(503)
            return httpx.Response(200, json=response)

        original_client = httpx.AsyncClient
        mock_transport = httpx.MockTransport(handler)
        with patch.object(main, "GEMINI_API_KEY", "test-key"), patch.object(
            main.asyncio, "sleep", new=AsyncMock()
        ):
            with patch.object(
                main.httpx,
                "AsyncClient",
                side_effect=lambda **kwargs: original_client(transport=mock_transport, **kwargs),
            ):
                result = asyncio.run(main.extract_once([(1, b"jpeg-bytes")], "test prompt"))

        self.assertEqual(result, {"first_name": "Maya"})
        self.assertEqual(attempts, 2)

    def test_persistent_gemini_503_reports_temporary_unavailability(self) -> None:
        attempts = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(503)

        original_client = httpx.AsyncClient
        mock_transport = httpx.MockTransport(handler)
        with patch.object(main, "GEMINI_API_KEY", "test-key"), patch.object(
            main.asyncio, "sleep", new=AsyncMock()
        ):
            with patch.object(
                main.httpx,
                "AsyncClient",
                side_effect=lambda **kwargs: original_client(transport=mock_transport, **kwargs),
            ):
                with self.assertRaises(main.HTTPException) as context:
                    asyncio.run(main.extract_once([(1, b"jpeg-bytes")], "test prompt"))

        self.assertEqual(context.exception.status_code, 502)
        self.assertEqual(attempts, main.GEMINI_MAX_ATTEMPTS)
        self.assertIn("remained unavailable (HTTP 503)", context.exception.detail)

    def test_upload_requires_external_processing_consent(self) -> None:
        with self.authenticated_client() as client:
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

    def test_pdf_extracts_each_page_individually_into_records(self) -> None:
        page_count = 11
        document = pymupdf.open()
        for _ in range(page_count):
            document.new_page(width=72, height=72)
        pdf_bytes = document.tobytes()
        document.close()

        async def extract_page(pages: list[tuple[int, bytes]]) -> dict:
            page_number = pages[0][0]
            return {
                "first_name": "Maya",
                "last_name": "Cohen",
                "id_number": "123456782",
                "central_numbers": [f"C-{page_number}"],
                "received_date": "12/04/2025",
                "referring_entity": "Clalit",
                "authorizations": [{
                    "number": f"A-{page_number}",
                    "label": "commitment",
                    "confidence": 0.95,
                    "evidence": f"Page {page_number} commitment",
                }],
                "field_confidence": {
                    "first_name": 0.95,
                    "last_name": 0.95,
                    "id_number": 0.95,
                    "central_numbers": 0.95,
                    "received_date": 0.95,
                    "referring_entity": 0.95,
                },
                "evidence": {
                    "first_name": "Maya",
                    "last_name": "Cohen",
                    "id_number": "123456782",
                    "central_numbers": f"Central {page_number}",
                    "received_date": "Received stamp",
                    "referring_entity": "Clalit logo",
                },
                "notes": [],
            }

        extractor = AsyncMock(side_effect=extract_page)
        with patch.object(main, "extract_document", new=extractor):
            with self.authenticated_client() as client:
                response = client.post(
                    "/api/extract",
                    data={"external_processing_consent": "true"},
                    files=[("files", ("commitments.pdf", pdf_bytes, "application/pdf"))],
                )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["errors"], [])
        self.assertEqual(len(result["documents"]), 1)
        records = result["documents"][0]["records"]
        self.assertEqual(len(records), page_count)
        self.assertEqual(extractor.await_count, page_count)
        self.assertEqual([record["source_pages"] for record in records], [[page] for page in range(1, page_count + 1)])
        self.assertEqual(
            [record["authorization_number"] for record in records],
            [f"A-{page}" for page in range(1, page_count + 1)],
        )

    def test_extract_document_runs_barcode_decoder_in_worker_thread(self) -> None:
        extracted = {
            "first_name": "Maya",
            "last_name": "Cohen",
            "id_number": "123456782",
            "central_numbers": ["937309"],
            "received_date": "12/04/2025",
            "referring_entity": "Clalit",
            "authorizations": [],
            "field_confidence": {
                "first_name": 0.9,
                "last_name": 0.9,
                "id_number": 0.9,
                "central_numbers": 0.9,
                "received_date": 0.9,
                "referring_entity": 0.9,
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
        thread_ids = []

        def decode(_image: bytes) -> list[dict]:
            thread_ids.append(threading.get_ident())
            return [{"text": "937309", "format": "Code 128", "rotation": 0, "bbox": {}}]

        with patch.object(main, "extract_once", new=AsyncMock(return_value=extracted)):
            with patch.object(main, "decode_page_barcodes", side_effect=decode):
                result = asyncio.run(main.extract_document([(1, b"page-image")]))

        self.assertEqual(result["decoded_central_barcodes"][0]["text"], "937309")
        self.assertNotEqual(thread_ids[0], threading.get_ident())


if __name__ == "__main__":
    unittest.main()
