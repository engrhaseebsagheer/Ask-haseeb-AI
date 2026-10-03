import os
from typing import Dict, List

from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

# Talks to the Drive REST API directly, so the large google-api-python-client package is not needed.
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
API = "https://www.googleapis.com/drive/v3/files"


def _session() -> AuthorizedSession:
    """Authenticate with the Google Drive API using the service-account JSON file."""
    if not GOOGLE_SERVICE_ACCOUNT_JSON or not os.path.exists(GOOGLE_SERVICE_ACCOUNT_JSON):
        raise FileNotFoundError(f"Google service account JSON not found: {GOOGLE_SERVICE_ACCOUNT_JSON}")
    creds = service_account.Credentials.from_service_account_file(GOOGLE_SERVICE_ACCOUNT_JSON, scopes=SCOPES)
    return AuthorizedSession(creds)


def list_files_in_folder(folder_id: str) -> List[Dict]:
    """List all files in the Drive folder."""
    session = _session()
    files: List[Dict] = []
    params = {
        "q": f"'{folder_id}' in parents and trashed=false",
        "pageSize": 1000,
        "fields": "nextPageToken, files(id, name, mimeType, modifiedTime)",
    }
    while True:
        resp = session.get(API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        files.extend(data.get("files", []))
        if not data.get("nextPageToken"):
            return files
        params["pageToken"] = data["nextPageToken"]


def download_file(file_id: str, name: str, mime_type: str, dest_path: str) -> str:
    """Download a file from Drive. Google Docs have no file to download, so their text is exported."""
    session = _session()
    if mime_type == "application/vnd.google-apps.document":
        resp = session.get(f"{API}/{file_id}/export", params={"mimeType": "text/plain"}, timeout=60)
        if not dest_path.lower().endswith(".txt"):
            dest_path += ".txt"
    else:
        resp = session.get(f"{API}/{file_id}", params={"alt": "media"}, timeout=60)
    resp.raise_for_status()

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    return dest_path
