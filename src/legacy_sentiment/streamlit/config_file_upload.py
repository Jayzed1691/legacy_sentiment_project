"""Streamlit helpers for uploading configuration files."""

from __future__ import annotations

from typing import List

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile


def upload_custom_files(label: str = "Upload configuration files") -> List[UploadedFile]:
    """Render a sidebar uploader for configuration assets."""
    return st.sidebar.file_uploader(
        label,
        type=["json"],
        accept_multiple_files=True,
        help="Upload one or more JSON configuration files",
    ) or []
