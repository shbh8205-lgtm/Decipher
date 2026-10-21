# Decipher

Decipher turns authorization-form scans and photos into editable lab records. It analyzes each PDF
page separately or groups explicitly selected image pages into one form, extracts visible values
with Gemini, validates common fields, and flags records that need human review. Each distinct
authorization becomes a record; shared patient fields are repeated on each record.

## Configure and run locally

1. Install Python 3.11 or newer.
2. Create a Gemini API key in Google AI Studio and confirm that your organization's privacy,
   consent, and regulatory requirements permit sending these medical documents to Google's API.
   NetFree filtering must allow the app's server to reach `generativelanguage.googleapis.com`.
3. Set the key and, optionally, the model in the PowerShell session where the app will run:

   ```powershell
   $env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
   $env:GEMINI_MODEL = "gemini-3.8-flash"
   $env:SESSION_SECRET = (python -c "import secrets; print(secrets.token_urlsafe(32))")
   $env:APP_USERS = 'operator:$2b$12$PASTE_BCRYPT_HASH_HERE'
   ```

   The application reads environment variables; it does not load `.env` automatically. The API
   key must remain in the server environment and must never be added to browser code, committed,
   or shared. Replace the example account hash with one generated as described in the deployment
   section. `.env.example` lists supported settings.
4. Create a virtual environment and install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

5. Start the app:

   ```powershell
   uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

6. Open <http://127.0.0.1:8000>.

## Processing and review

- Supported inputs: PDF, PNG, JPEG, TIFF, BMP, and WebP; maximum 20 MB and 12 pages per form.
  Use the grouping option to treat selected images as pages from one photographed/scanned form;
  PDFs already group their pages automatically.
- PDFs are rendered at 250 DPI; image orientation is corrected and images are normalized before
  extraction. PDF pages are extracted individually, and the source page number is stored with
  each resulting authorization record. Image pages selected with the grouping option are treated
  together as one form.
- Central-number barcodes are decoded locally with ZXing after the two Gemini readings are
  reconciled. Barcode images and crops are not saved or sent to Gemini; the database stores only
  the decoded value, verification status, and any OCR alternative. A missing barcode is normal.
  Multiple barcode candidates are not chosen blindly: a candidate is accepted only when it matches
  an extracted central number, except for a single unambiguous barcode-only reading.
- The central-number field shows barcode/OCR match, barcode-only, OCR-only, conflict, or manual
  correction status. Conflicts use the barcode value and retain the model reading as an alternative.
  Barcode-only and conflict results require review; OCR-only results do not require review merely
  because a barcode was absent. The CSV export includes a **Central Verification** column.
- Each form is read twice independently. Disagreements are surfaced as review notes; number
  candidates found in either pass are retained rather than silently discarded.
- Uploads are disabled until an operator confirms authorization for external Gemini processing.
  The server also enforces this confirmation on the extraction endpoint.
- The Gemini integration uses the Interactions API, sends `store: false` to avoid retaining
  interaction history there, and requests JSON matching the extraction schema. Set
  `GEMINI_MODEL` to an available model ID, such as `gemini-3.8-flash`.
- Temporary Gemini HTTP 503 responses are retried up to three times with a short backoff.
  Persistent 503 errors are reported as temporary service unavailability; wait and retry, then
  check Google's service status and network filtering if the issue continues.
- If extraction reports HTTP 404, confirm `GEMINI_MODEL` is an available model ID and that NetFree
  allows the server to access `generativelanguage.googleapis.com` without replacing/blocking the
  Gemini API response. The application shows Gemini's returned error message when available.
- Document images are transmitted to the Google Gemini API using the server-side API key. Do not
  upload real patient forms unless authorized, and verify applicable Google API terms, data
  handling/retention settings, and any required organizational agreements before use. API keys
  are credentials: keep them secret, restrict their permissions/quotas in Google AI Studio, and
  rotate exposed keys.
- Uploaded originals are processed in memory by the application and are not stored on the server.
  A small first-page thumbnail is saved with each record in the local database so the list and
  record preview still work after reload. Original full-resolution documents are not retained;
  the browser's full-document preview is available only during the upload session.
- Structured records are stored in `data/decipher.sqlite3` (or `DATA_DIR`). This database contains
  medical and identity information. Protect the host and database with appropriate access controls
  and encrypted storage.
- Low model confidence (< 0.75), missing values, invalid Israeli ID check digits, implausible
  receipt dates (before 2000 or in the future), unknown entities, and model notes require review.
  Confidence is a model-provided signal, not a measured accuracy guarantee.
- The UI lets an operator correct fields and approve a record. An approval does not remove its
  review reasons; it records the operator's decision.
- Use **Export CSV** above the records list or visit `/api/records.csv` to download all saved
  authorization records. The UTF-8 CSV includes review reasons and source filename/pages; ID
  values are emitted in an Excel-compatible text form to preserve leading zeroes.
- To measure local barcode coverage on sample PDFs, run
  `python scripts/barcode_report.py path\to\sample.pdf` or pass a directory. The report renders
  pages at the same 250 DPI as the application and compares four-rotation decoding with contrast,
  1.5x/2x upscaling, and top/bottom band-crop fallbacks.

## Evaluation

Create a model-prefilled draft from PDFs, then manually verify every cell against its `page_image`;
the draft is **not ground truth** until a person has checked and corrected it. Add any missed
authorizations and remove any spurious rows, then copy the verified rows into the local,
git-ignored `eval/ground_truth.csv`. Use one row per authorization, set
`split` to `dev` for development or `test` for final, held-out measurement, and leave a field
empty only when it is genuinely absent or unreadable on the form.

```powershell
python scripts/make_gt_draft.py path\to\sample.pdf
python scripts/make_gt_draft.py path\to\pdf-folder --split dev
python scripts/evaluate.py --split dev --cache
python scripts/evaluate.py --split test --from-db
python scripts/evaluate.py --split all --live
```

`--live` uses the local rendered page images and the configured Gemini API, saving raw per-page
responses to `eval/cache/`; `--cache` evaluates matching cached responses without API calls
(the rendered images from draft generation must remain available); `--from-db` compares saved
records without changing the database schema. Results are written to
`eval/report.md` and `eval/mismatches.csv`. Ground truth, page images, response cache, draft, and
mismatch details are git-ignored because they may contain names and identity information. Accuracy
figures are measurements on the team's own manually labeled sample, not a guarantee of performance
on future forms. Keep the `test` split held out from prompt tuning and use it only for final numbers.

## Render deployment

The repository includes [render.yaml](./render.yaml). Connect the repository to Render and deploy
the Blueprint; it provisions a persistent disk mounted at `/var/data` and starts Uvicorn on
`0.0.0.0:$PORT`. SQLite records persist across redeploys only while `DATA_DIR` points inside that
mounted disk. The Blueprint sets `DATA_DIR=/var/data`. Keep the service at one instance when using
SQLite; this deployment does not provide shared multi-instance database access.

`GET /healthz` is a public, lightweight uptime endpoint returning only `{"ok":true}`. It does not
check Gemini or read the database. Render uses it for its health check. Free Render web services
sleep after 15 minutes without inbound traffic; their local filesystem, including SQLite, is
ephemeral and can be wiped on restart/redeploy. A persistent disk requires a plan that supports
disks; the Blueprint's disk is intended for that deployment. `DEMO_SEED=1` inserts a few clearly
synthetic sample records only when the database is empty at startup; set it to `0` to disable this.

For an external uptime monitor, create an HTTP(S) check in UptimeRobot or cron-job.org pointing to
`https://<your-render-service>.onrender.com/healthz`, use **GET**, and set the interval to
**10 minutes**. No login or API key is needed for this endpoint. A ping may wake a sleeping free
instance and incur its cold-start delay; do not rely on external pings to prevent free-tier sleep
or to preserve ephemeral SQLite data.

Set these environment variables in Render:

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Server-side Gemini credential |
| `GEMINI_MODEL` | Yes | Model ID, e.g. `gemini-3.8-flash` |
| `SESSION_SECRET` | Yes | Long random signing key for login sessions; Render can generate it |
| `APP_USERS` | Yes, or use `APP_PASSWORD_HASH` | Comma-separated `username:bcrypt_hash` accounts; no self-registration |
| `DATA_DIR` | Yes | Persistent disk path, `/var/data` in the Blueprint |
| `DEMO_BANNER` | No | Set `true` to show “Demo environment - sample data only” |
| `DEMO_SEED` | No | Set `1` to populate an empty database with synthetic demo rows |

Generate an account password hash locally (never place a plaintext password in the deployment
configuration):

```powershell
python -c "import bcrypt; print(bcrypt.hashpw(input('Password: ').encode(), bcrypt.gensalt()).decode())"
```

Then set `APP_USERS` to `operator:<generated-hash>` in Render. Alternatively set
`APP_PASSWORD_HASH` to a bcrypt hash; this enables the single username `operator`. The
`APP_PASSWORD_HASH` option is intended for a single account and is used only when `APP_USERS` is
not configured.

Login sessions use signed, HttpOnly, Secure, SameSite=Lax cookies. `/` and every `/api/*` route
require authentication; `/healthz`, `/login`, `/logout`, and static assets are intentionally
public. Login and extraction attempts are rate-limited in memory per application process. CORS is
not enabled, and responses include a restrictive Content Security Policy, `X-Frame-Options`, and
`X-Content-Type-Options`. Do not log uploaded document contents or extracted patient values.

## Limitations

There is no labeled benchmark in this workspace, so extraction accuracy is not claimed. Review
and correct records against the project's ground truth before using it for operational decisions.
The hospital/entity list and date plausibility limits should be tuned against the lab's actual
forms. No laboratory API integration is included.

**Not production-ready for real patient data.** This deployment configuration is intended for a
controlled demo or evaluation, not as a certification of production security or regulatory
compliance. In particular:

- There is no user provisioning/registration, per-user role model, MFA, password reset, or audit log.
- Login and extraction rate limits are process-local and reset on restart; they are not a shared
  abuse-prevention system.
- SQLite on a Render disk supports a single service instance and does not provide managed database
  backups, replication, or high availability.
- Application login does not replace TLS, network restrictions, disk encryption, secret rotation,
  incident response, retention policy, or organizational privacy/security review.
- Uploaded documents are sent to Gemini; verify applicable consent, provider terms, and legal/
  regulatory requirements before processing any real patient information.
- The demo banner is informational only and does not sanitize, anonymize, or otherwise protect data. is included in this initial application.
