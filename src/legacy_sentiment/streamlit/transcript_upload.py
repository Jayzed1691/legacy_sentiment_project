"""Streamlit widgets for uploading transcript documents."""

from __future__ import annotations

from typing import List

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile


def upload_transcript_files(label: str = "Upload transcripts") -> List[UploadedFile]:
    """Render a sidebar uploader for transcript documents."""
    return st.sidebar.file_uploader(
        label,
        type=["json", "txt", "pdf"],
        accept_multiple_files=True,
        help="Supported formats: JSON, TXT, PDF",
    ) or []
