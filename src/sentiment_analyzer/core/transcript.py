"""
Transcript Parser

Unified parser for earnings call transcripts in multiple formats (JSON, TXT, PDF).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sentiment_analyzer.data.models import DialogueEntry, Section, TranscriptData

logger = logging.getLogger(__name__)


class TranscriptParserError(Exception):
    """Custom exception for transcript parsing errors."""
    pass


class TranscriptParser:
    """
    Unified parser for earnings call transcripts.

    Supports multiple formats:
    - JSON: Structured transcript with sections and speakers
    - TXT: Plain text with section markers and speaker labels
    - PDF: PDF transcripts (requires PyPDF2 or similar)

    Example:
        >>> parser = TranscriptParser()
        >>> transcript = parser.parse_file('earnings_call.json', 'json')
        >>> for section in transcript.sections:
        ...     print(f"Section: {section.name}")
        ...     for dialogue in section.dialogues:
        ...         print(f"  {dialogue.speaker}: {dialogue.text[:50]}...")
    """

    def __init__(self):
        """Initialize the transcript parser."""
        pass

    def parse_file(self, file_path: Union[str, Path], file_type: Optional[str] = None) -> TranscriptData:
        """
        Parse a transcript file.

        Args:
            file_path: Path to the transcript file
            file_type: File type ('json', 'txt', 'pdf'). Auto-detected if None.

        Returns:
            TranscriptData object containing parsed transcript

        Raises:
            TranscriptParserError: If file cannot be parsed
        """
        file_path = Path(file_path)

        # Auto-detect file type if not provided
        if file_type is None:
            file_type = file_path.suffix.lstrip('.').lower()

        # Read file content
        content = self._read_file(file_path, file_type)

        # Parse based on type
        return self.parse_content(content, file_type)

    def parse_content(self, content: Any, file_type: str) -> TranscriptData:
        """
        Parse transcript content.

        Args:
            content: File content (string for JSON/TXT, bytes for PDF)
            file_type: File type ('json', 'txt', 'pdf')

        Returns:
            TranscriptData object

        Raises:
            TranscriptParserError: If content cannot be parsed
        """
        file_type = file_type.lower()

        try:
            if file_type == 'json':
                return self._parse_json(content)
            elif file_type == 'txt':
                return self._parse_txt(content)
            elif file_type == 'pdf':
                return self._parse_pdf(content)
            else:
                raise TranscriptParserError(f"Unsupported file type: {file_type}")
        except Exception as e:
            logger.error(f"Error parsing {file_type} content: {e}")
            raise TranscriptParserError(f"Failed to parse {file_type}: {e}") from e

    def format_transcript(self, transcript: TranscriptData, indent_level: int = 0) -> str:
        """
        Format transcript data as a readable string.

        Args:
            transcript: TranscriptData to format
            indent_level: Starting indentation level

        Returns:
            Formatted string representation
        """
        return ''.join(self._format_section(section, indent_level) for section in transcript.sections)

    # =========================================================================
    # JSON Parsing
    # =========================================================================

    def _parse_json(self, content: Union[str, bytes, Dict]) -> TranscriptData:
        """Parse JSON transcript."""
        # Convert to dict if string/bytes
        if isinstance(content, (bytes, bytearray)):
            content = content.decode('utf-8')
        if isinstance(content, str):
            content = json.loads(content)

        transcript_data = TranscriptData()

        # Extract metadata if present
        if 'metadata' in content:
            transcript_data.metadata = content['metadata']

        # Parse sections
        transcript_list = content.get('transcript', [])
        for section_data in transcript_list:
            section = self._parse_json_section(section_data)
            transcript_data.sections.append(section)

        # Build speaker index
        self._build_speaker_index(transcript_data)

        logger.info(f"Parsed JSON transcript: {len(transcript_data.sections)} sections, "
                   f"{len(transcript_data.speakers)} speakers")

        return transcript_data

    def _parse_json_section(self, section_data: Dict[str, Any]) -> Section:
        """Parse a JSON section."""
        section_name = section_data.get('section', 'Unnamed Section')
        section = Section(name=section_name)

        # Parse speakers in the current section
        for speaker_data in section_data.get('speakers', []):
            dialogue_entry = DialogueEntry(
                speaker=speaker_data['name'],
                role=speaker_data.get('role', ''),
                text=speaker_data['dialogue']
            )
            section.dialogues.append(dialogue_entry)

        # Parse subsections recursively
        for subsection_data in section_data.get('subsections', []):
            subsection = self._parse_json_section(subsection_data)
            section.subsections.append(subsection)

        return section

    # =========================================================================
    # TXT Parsing
    # =========================================================================

    def _parse_txt(self, content: Union[str, bytes]) -> TranscriptData:
        """Parse plain text transcript."""
        # Convert to string if bytes
        if isinstance(content, (bytes, bytearray)):
            content = content.decode('utf-8')

        transcript_data = TranscriptData()
        lines = content.splitlines()

        i = 0
        has_sections = any(line.strip().startswith("Section:") for line in lines)

        # If no explicit sections, create a default section for all dialogues
        if not has_sections:
            default_section = Section(name="Transcript")
            transcript_data.sections.append(default_section)
            self._parse_txt_section(lines, 0, default_section)
        else:
            # Parse with explicit sections
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("Section:"):
                    section_name = line.split("Section:")[1].strip()
                    section = Section(name=section_name)
                    transcript_data.sections.append(section)
                    i = self._parse_txt_section(lines, i + 1, section)
                else:
                    i += 1

        # Build speaker index
        self._build_speaker_index(transcript_data)

        logger.info(f"Parsed TXT transcript: {len(transcript_data.sections)} sections, "
                   f"{len(transcript_data.speakers)} speakers")

        return transcript_data

    def _parse_txt_section(self, lines: List[str], start_index: int, section: Section) -> int:
        """Parse a TXT section."""
        i = start_index
        current_subsection = None

        while i < len(lines):
            line = lines[i].strip()

            # New section starts - return to parent
            if line.startswith("Section:"):
                return i

            # Subsection
            elif line.startswith("Sub-Section:"):
                subsection_name = line.split("Sub-Section:")[1].strip()
                current_subsection = Section(name=subsection_name)
                section.subsections.append(current_subsection)

            # Dialogue line (contains colon)
            elif ":" in line:
                try:
                    speaker_part, dialogue = line.split(":", 1)
                    speaker, role = self._extract_speaker_and_role(speaker_part)
                    dialogue_entry = DialogueEntry(
                        speaker=speaker.strip(),
                        role=role,
                        text=dialogue.strip()
                    )

                    # Add to subsection or main section
                    if current_subsection:
                        current_subsection.dialogues.append(dialogue_entry)
                    else:
                        section.dialogues.append(dialogue_entry)

                except ValueError:
                    # Line has colon but isn't dialogue - append to previous dialogue
                    self._append_to_last_dialogue(section, current_subsection, line)

            # Continuation line
            else:
                self._append_to_last_dialogue(section, current_subsection, line)

            i += 1

        return i

    def _extract_speaker_and_role(self, speaker_text: str) -> Tuple[str, str]:
        """Extract speaker name and role from text like 'John Smith (CEO)'."""
        if "(" in speaker_text and ")" in speaker_text:
            name, role = speaker_text.split("(", 1)
            role = role.rstrip(")").strip()
            return name.strip(), role
        return speaker_text.strip(), ""

    def _append_to_last_dialogue(self, section: Section, subsection: Optional[Section], text: str) -> None:
        """Append text to the last dialogue entry."""
        if subsection and subsection.dialogues:
            subsection.dialogues[-1].text += " " + text
        elif section.dialogues:
            section.dialogues[-1].text += " " + text

    # =========================================================================
    # PDF Parsing
    # =========================================================================

    def _parse_pdf(self, content: bytes) -> TranscriptData:
        """
        Parse PDF transcript.

        Note: This is a placeholder. PDF parsing requires additional dependencies
        like PyPDF2 or pdfplumber.
        """
        try:
            import PyPDF2
            from io import BytesIO

            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Parse extracted text as plain text
            return self._parse_txt(text)

        except ImportError:
            logger.warning("PyPDF2 not installed. Cannot parse PDF files.")
            raise TranscriptParserError("PDF parsing requires PyPDF2. Install with: pip install PyPDF2")

    # =========================================================================
    # Utilities
    # =========================================================================

    def _read_file(self, file_path: Path, file_type: str) -> Any:
        """Read file content."""
        mode = 'rb' if file_type.lower() == 'pdf' else 'r'

        try:
            with open(file_path, mode) as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise TranscriptParserError(f"Failed to read file: {e}") from e

    def _build_speaker_index(self, transcript: TranscriptData) -> None:
        """Build index of speakers to their dialogue texts."""
        for dialogue in transcript.all_dialogues():
            transcript.speakers[dialogue.speaker].append(dialogue.text)

    def _format_section(self, section: Section, indent_level: int) -> str:
        """Format a section as a string."""
        indent = '  ' * indent_level
        result = indent + f"Section: {section.name}\n"

        # Format dialogues
        for dialogue in section.dialogues:
            role_display = f" ({dialogue.role})" if dialogue.role else ''
            result += f"{indent}  {dialogue.speaker}{role_display}: {dialogue.text}\n"

        # Format subsections recursively
        for subsection in section.subsections:
            result += self._format_section(subsection, indent_level + 1)

        return result


# =============================================================================
# Convenience Functions
# =============================================================================

def parse_transcript(file_path: Union[str, Path], file_type: Optional[str] = None) -> TranscriptData:
    """
    Convenience function to parse a transcript file.

    Args:
        file_path: Path to transcript file
        file_type: File type ('json', 'txt', 'pdf'). Auto-detected if None.

    Returns:
        TranscriptData object
    """
    parser = TranscriptParser()
    return parser.parse_file(file_path, file_type)


def format_transcript(transcript: TranscriptData) -> str:
    """
    Convenience function to format a transcript.

    Args:
        transcript: TranscriptData to format

    Returns:
        Formatted string
    """
    parser = TranscriptParser()
    return parser.format_transcript(transcript)
