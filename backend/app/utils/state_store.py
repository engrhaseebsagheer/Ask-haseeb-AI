import json
import os
from typing import Dict

STATE_PATH = "backend/data/processed/processed_files.json"

def load_state() -> Dict[str, str]:
    if not os.path.exists(STATE_PATH):
        return {}
    with open(STATE_PATH, "r") as f:
        return json.load(f)

def save_state(state: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
