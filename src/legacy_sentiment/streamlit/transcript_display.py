"""Streamlit rendering helpers for transcript structures."""

from __future__ import annotations

from typing import Iterable, Tuple

import streamlit as st

from legacy_sentiment.data_models.transcript_structures import Section, TranscriptData


def display_transcripts(transcripts: Iterable[Tuple[TranscriptData, str]], nlp=None) -> None:
    """Render transcripts and their sections inside expandable containers."""
    for transcript, title in transcripts:
        with st.expander(title, expanded=False):
            for section in transcript.sections:
                _render_section(section, 0)


def _render_section(section: Section, indent: int) -> None:
    header = f"{' ' * (indent * 2)}• **{section.name}**"
    st.markdown(header)
    for dialogue in section.dialogues:
        role_suffix = f" ({dialogue.role})" if dialogue.role else ""
        st.markdown(
            f"{' ' * ((indent + 1) * 2)}- **{dialogue.speaker}{role_suffix}:** {dialogue.text}"
        )
    for subsection in section.subsections:
        _render_section(subsection, indent + 1)
