"""Utilities for parsing uploaded configuration files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile


def load_configs(files: Iterable[UploadedFile]) -> Dict[str, dict]:
    """Load JSON configuration data keyed by file stem."""
    configs: Dict[str, dict] = {}
    for uploaded in files or []:
        try:
            config = json.load(uploaded)
            uploaded.seek(0)
        except json.JSONDecodeError as exc:
            st.warning(f"Failed to parse {uploaded.name}: {exc}")
            uploaded.seek(0)
            continue
        key = Path(uploaded.name).stem
        configs[key] = config
    return configs
