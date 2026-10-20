# Decipher

Decipher turns authorization-form scans and photos into editable lab records. It groups PDF pages
or explicitly selected image pages into one form, extracts visible values with Gemini, validates
common fields, and flags records that need human review. Each distinct authorization becomes a
record; the document's central-number list and shared patient fields are repeated on each record.

## Configure and run locally

1. Install Python 3.11 or newer.
2. Create a Gemini API key in Google AI Studio and confirm that your organization's privacy,
   consent, and regulatory requirements permit sending these medical documents to Google's API.
   NetFree filtering must allow the app's server to reach `generativelanguage.googleapis.com`.
3. Set the key and, optionally, the model in the PowerShell session where the app will run:

   ```powershell
   $env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
   $env:GEMINI_MODEL = "gemini-3.8-flash"
   ```

   The application reads environment variables; it does not load `.env` automatically. The API
   key must remain in the server environment and must never be added to browser code, committed,
   or shared. `.env.example` lists supported settings.
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
  extraction. All pages of one form are sent to Gemini together.
- Each form is read twice independently. Disagreements are surfaced as review notes; number
  candidates found in either pass are retained rather than silently discarded.
- Uploads are disabled until an operator confirms authorization for external Gemini processing.
  The server also enforces this confirmation on the extraction endpoint.
- The Gemini integration uses the Interactions API, sends `store: false` to avoid retaining
  interaction history there, and requests JSON matching the extraction schema. Set
  `GEMINI_MODEL` to an available model ID, such as `gemini-3.8-flash`.
- If extraction reports HTTP 404, confirm `GEMINI_MODEL` is an available model ID and that NetFree
  allows the server to access `generativelanguage.googleapis.com` without replacing/blocking the
  Gemini API response. The application shows Gemini's returned error message when available.
- Document images are transmitted to the Google Gemini API using the server-side API key. Do not
  upload real patient forms unless authorized, and verify applicable Google API terms, data
  handling/retention settings, and any required organizational agreements before use. API keys
  are credentials: keep them secret, restrict their permissions/quotas in Google AI Studio, and
  rotate exposed keys.
- Uploaded originals are processed in memory by the application and are not stored on the server.
  The browser keeps a local preview only for the current page session.
- Structured records are stored in `data/decipher.sqlite3` (or `DATA_DIR`). This database contains
  medical and identity information. Protect the host and database with appropriate access controls
  and encrypted storage; this local prototype has no user authentication and is not suitable for
  network exposure or production deployment.
- Low model confidence (< 0.75), missing values, invalid Israeli ID check digits, implausible
  receipt dates (before 2000 or in the future), unknown entities, and model notes require review.
  Confidence is a model-provided signal, not a measured accuracy guarantee.
- The UI lets an operator correct fields and approve a record. An approval does not remove its
  review reasons; it records the operator's decision.

## Limitations

There is no labeled benchmark in this workspace, so extraction accuracy is not claimed. Review
and correct records against the project's ground truth before using it for operational decisions.
The hospital/entity list and date plausibility limits should be tuned against the lab's actual
forms. No Excel export or laboratory API integration is included in this initial application.
