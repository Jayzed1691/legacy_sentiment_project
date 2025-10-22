#!/usr/bin/env python3

"""Transcript handling utilities used by the Streamlit front-end."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, Sequence, Tuple

from legacy_sentiment.data_models.transcript_structures import Section, TranscriptData
from legacy_sentiment.ingestion.json_transcript_parser import JSONTranscriptParser
from legacy_sentiment.ingestion.pdf_transcript_parser import PDFTranscriptParser
from legacy_sentiment.ingestion.txt_transcript_parser import TXTTranscriptParser

if TYPE_CHECKING:  # pragma: no cover - optional Streamlit dependency
        from streamlit.runtime.uploaded_file_manager import UploadedFile


class TranscriptHandler:
        """High level API for parsing and formatting transcript documents."""

        def __init__(self) -> None:
                self._json_parser = JSONTranscriptParser()
                self._txt_parser = TXTTranscriptParser()
                self._pdf_parser = PDFTranscriptParser()

        def parse_file(self, file_content: Any, file_type: str) -> TranscriptData:
                """Parse raw file content into :class:`TranscriptData`."""
                normalized_type = file_type.lower()
                if normalized_type == 'json':
                        data = self._ensure_json_data(file_content)
                        return self._json_parser.parse(data)
                if normalized_type == 'txt':
                        text = self._ensure_text(file_content)
                        return self._txt_parser.parse(text)
                if normalized_type == 'pdf':
                        return self._pdf_parser.parse(file_content)
                raise ValueError(f"Unsupported file type: {file_type}")

        def process_transcript(self, file_path: str, file_type: str) -> Tuple[TranscriptData, str]:
                        file_content = self.read_file(file_path, file_type)
                        transcript_data = self.parse_file(file_content, file_type)
                        formatted_transcript = self.format_transcript(transcript_data)
                        return transcript_data, formatted_transcript

        def process_multiple_transcripts(self, file_paths: Sequence[str], file_types: Sequence[str]) -> List[Tuple[TranscriptData, str]]:
                results: List[Tuple[TranscriptData, str]] = []
                for file_path, file_type in zip(file_paths, file_types):
                        try:
                                results.append(self.process_transcript(file_path, file_type))
                        except Exception as exc:  # pragma: no cover - defensive logging
                                print(f"Error processing file {file_path}: {exc}")
                return results

        def parse_files(self, files: Iterable['UploadedFile']) -> List[TranscriptData]:
                """Parse uploaded Streamlit files into :class:`TranscriptData` objects."""
                transcripts: List[TranscriptData] = []
                for uploaded in files:
                        file_type = Path(uploaded.name).suffix.lstrip('.').lower()
                        raw_bytes = uploaded.read()
                        uploaded.seek(0)
                        content: Any
                        if file_type == 'json':
                                content = raw_bytes.decode('utf-8')
                        elif file_type == 'txt':
                                content = raw_bytes.decode('utf-8')
                        else:
                                content = raw_bytes
                        transcripts.append(self.parse_file(content, file_type))
                return transcripts

        def read_file(self, file_path: str, file_type: str) -> Any:
                mode = 'rb' if file_type.lower() == 'pdf' else 'r'
                with open(file_path, mode) as file:
                        return file.read()

        def format_transcript(self, transcript_data: TranscriptData, indent_level: int = 0) -> str:
                return ''.join(self._format_section(section, indent_level) for section in transcript_data.sections)

        def _format_section(self, section: Section, indent_level: int) -> str:
                indent = '  ' * indent_level
                section_str = indent + f"Section: {section.name}\n"

                for dialogue in section.dialogues:
                        role_display = f" ({dialogue.role})" if dialogue.role else ''
                        section_str += f"{indent}  {dialogue.speaker}{role_display}: {dialogue.text}\n"

                for subsection in section.subsections:
                        section_str += self._format_section(subsection, indent_level + 1)

                return section_str

        def _ensure_json_data(self, file_content: Any) -> Any:
                if isinstance(file_content, (bytes, bytearray)):
                        file_content = file_content.decode('utf-8')
                if isinstance(file_content, str):
                        return json.loads(file_content)
                return file_content

        def _ensure_text(self, file_content: Any) -> str:
                if isinstance(file_content, (bytes, bytearray)):
                        return file_content.decode('utf-8')
                return str(file_content)


def parse_file(file_content: Any, file_type: str) -> TranscriptData:
        return TranscriptHandler().parse_file(file_content, file_type)


def get_full_transcript(transcript_data: TranscriptData, indent_level: int = 0) -> str:
        handler = TranscriptHandler()
        return handler.format_transcript(transcript_data, indent_level)


def process_transcript(file_path: str, file_type: str) -> Tuple[TranscriptData, str]:
        handler = TranscriptHandler()
        return handler.process_transcript(file_path, file_type)


def read_file(file_path: str, file_type: str) -> Any:
        handler = TranscriptHandler()
        return handler.read_file(file_path, file_type)


def process_multiple_transcripts(file_paths: Sequence[str], file_types: Sequence[str]) -> List[Tuple[TranscriptData, str]]:
        handler = TranscriptHandler()
        return handler.process_multiple_transcripts(file_paths, file_types)