import os
import io
from typing import List, Dict
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Load environment variables
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def _get_drive_service():
    """Authenticate with Google Drive API using service account JSON."""
    if not GOOGLE_SERVICE_ACCOUNT_JSON or not os.path.exists(GOOGLE_SERVICE_ACCOUNT_JSON):
        raise FileNotFoundError(f"Google service account JSON not found: {GOOGLE_SERVICE_ACCOUNT_JSON}")

    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_SERVICE_ACCOUNT_JSON, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)

def list_files_in_folder(folder_id: str) -> List[Dict]:
    """List all files in the specified Google Drive folder."""
    service = _get_drive_service()
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        pageSize=1000,
        fields="files(id, name, mimeType, modifiedTime)"
    ).execute()
    return results.get("files", [])

def download_file(file_id: str, name: str, mime_type: str, dest_path: str) -> str:
    """Download a file from Google Drive to local destination."""
    service = _get_drive_service()
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(fh.getvalue())

    return dest_path
