"""Stable paths shared by the server and offline CLI commands."""

from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = API_ROOT.parent.parent
PROFILE_PATH = PROJECT_ROOT / "profiles" / "default" / "profile.json"
