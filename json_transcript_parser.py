#!/usr/bin/env python3

# json_transcript_parser.py

import logging
from typing import Dict, List, Any
from dataclasses import dataclass, field
from collections import defaultdict
from transcript_structures import TranscriptData, Section, DialogueEntry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranscriptParserError(Exception):
	"""Custom exception for TranscriptParser errors."""
	pass
	
class JSONTranscriptParser:
	def parse(self, json_data: Dict[str, Any]) -> TranscriptData:
		transcript_data = TranscriptData()
		transcript_list = json_data.get('transcript', [])
		for section_data in transcript_list:
			section = self._parse_section(section_data)
			transcript_data.sections.append(section)
		return transcript_data
	
	def _parse_section(self, section_data: Dict[str, Any]) -> Section:
		section_name = section_data.get('section', 'Unnamed Section')
		section = Section(name=section_name)
		
		# Parse speakers in the current section
		for speaker_data in section_data.get('speakers', []):
			dialogue_entry = DialogueEntry(
				speaker=speaker_data['name'],
				role=speaker_data.get('role', 'Unknown Role'),
				text=speaker_data['dialogue']
			)
			section.dialogues.append(dialogue_entry)
			
		# Parse subsections
		for subsection_data in section_data.get('subsections', []):
			subsection = self._parse_section(subsection_data)
			section.subsections.append(subsection)
			
		return section