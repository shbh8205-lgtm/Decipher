import os
import unittest
from unittest.mock import patch

import bcrypt
from fastapi.testclient import TestClient

import app.main as main


class AuthenticationAndHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.password_hash = bcrypt.hashpw(
            b"correct-password",
            bcrypt.gensalt(),
        ).decode("ascii")
        self.original_app_users = os.environ.get("APP_USERS")
        self.original_app_password_hash = os.environ.get("APP_PASSWORD_HASH")
        os.environ["APP_USERS"] = f"lab:{self.password_hash}"
        os.environ.pop("APP_PASSWORD_HASH", None)
        main._RATE_BUCKETS.clear()
        self.client = TestClient(
            main.app,
            base_url="https://testserver",
            follow_redirects=False,
        )

    def tearDown(self) -> None:
        self.client.close()
        if self.original_app_users is None:
            os.environ.pop("APP_USERS", None)
        else:
            os.environ["APP_USERS"] = self.original_app_users
        if self.original_app_password_hash is None:
            os.environ.pop("APP_PASSWORD_HASH", None)
        else:
            os.environ["APP_PASSWORD_HASH"] = self.original_app_password_hash
        main._RATE_BUCKETS.clear()

    def test_unauthenticated_access_is_rejected_for_all_application_routes(self) -> None:
        health = self.client.get("/healthz")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"ok": True})
        self.assertEqual(self.client.get("/").status_code, 303)
        self.assertEqual(self.client.get("/").headers["location"], "/login")
        self.assertEqual(self.client.get("/api/status").status_code, 401)
        self.assertEqual(self.client.get("/api/records").status_code, 401)
        self.assertEqual(self.client.get("/api/records.csv").status_code, 401)
        self.assertEqual(self.client.post("/api/extract").status_code, 401)
        self.assertEqual(
            self.client.patch("/api/records/example", json={}).status_code,
            401,
        )
        self.assertEqual(self.client.get("/api/unregistered-route").status_code, 401)

    def test_health_check_does_not_access_database_or_gemini(self) -> None:
        with self.client:
            with patch.object(main, "connect_db", side_effect=AssertionError("database access")):
                with patch.object(main, "extract_once", side_effect=AssertionError("Gemini access")):
                    response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})

    def test_login_success_sets_secure_session_and_logout_clears_it(self) -> None:
        response = self.client.post(
            "/login",
            data={"username": "lab", "password": "correct-password"},
        )

        self.assertEqual(response.status_code, 303)
        cookie = response.headers["set-cookie"]
        self.assertIn("httponly", cookie.casefold())
        self.assertIn("secure", cookie.casefold())
        self.assertIn("samesite=lax", cookie.casefold())
        self.assertEqual(self.client.get("/api/status").status_code, 200)
        self.assertEqual(self.client.post("/logout").status_code, 303)
        self.assertEqual(self.client.get("/api/status").status_code, 401)

    def test_login_failure_does_not_authenticate(self) -> None:
        response = self.client.post(
            "/login",
            data={"username": "lab", "password": "wrong-password"},
        )

        self.assertEqual(response.status_code, 401)
        self.assertIn("Invalid username or password", response.text)
        self.assertEqual(self.client.get("/api/status").status_code, 401)

    def test_single_password_hash_configures_operator_account(self) -> None:
        with patch.dict(os.environ, {"APP_USERS": "", "APP_PASSWORD_HASH": self.password_hash}):
            response = self.client.post(
                "/login",
                data={"username": "operator", "password": "correct-password"},
            )

        self.assertEqual(response.status_code, 303)

    def test_login_attempts_are_rate_limited(self) -> None:
        with patch.object(main, "LOGIN_LIMIT", 2):
            for _ in range(2):
                response = self.client.post(
                    "/login",
                    data={"username": "lab", "password": "wrong-password"},
                )
                self.assertEqual(response.status_code, 401)
            limited = self.client.post(
                "/login",
                data={"username": "lab", "password": "correct-password"},
            )

        self.assertEqual(limited.status_code, 429)

    def test_extract_endpoint_is_rate_limited(self) -> None:
        login = self.client.post(
            "/login",
            data={"username": "lab", "password": "correct-password"},
        )
        self.assertEqual(login.status_code, 303)
        with patch.object(main, "EXTRACT_LIMIT", 1):
            first = self.client.post("/api/extract")
            second = self.client.post("/api/extract")

        self.assertEqual(first.status_code, 422)
        self.assertEqual(second.status_code, 429)

    def test_security_headers_and_demo_banner(self) -> None:
        with patch.object(main, "DEMO_BANNER", True):
            response = self.client.get("/login")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertIn("Content-Security-Policy", response.headers)
        self.assertIn("script-src 'self'", response.headers["content-security-policy"])

        self.client.post(
            "/login",
            data={"username": "lab", "password": "correct-password"},
        )
        with patch.object(main, "DEMO_BANNER", True):
            home = self.client.get("/")
        self.assertIn("Demo environment - sample data only", home.text)

    def test_demo_seed_populates_empty_database_once(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            with patch.object(main, "DATA_DIR", data_dir), patch.object(
                main, "DATABASE_PATH", data_dir / "test.sqlite3"
            ), patch.dict(os.environ, {"DEMO_SEED": "1"}):
                main.startup()
                first_run = main.list_records()
                main.startup()
                second_run = main.list_records()

        self.assertEqual(len(first_run), 3)
        self.assertEqual(len(second_run), 3)
        self.assertTrue(all(record["source_filename"] == "DEMO SAMPLE ONLY" for record in second_run))
        self.assertTrue(all("Demo seed: sample-only data" in record["review_reasons"] for record in second_run))

    def test_demo_seed_is_opt_in(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            with patch.object(main, "DATA_DIR", data_dir), patch.object(
                main, "DATABASE_PATH", data_dir / "test.sqlite3"
            ), patch.dict(os.environ, {"DEMO_SEED": "0"}):
                main.startup()
                records = main.list_records()
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
