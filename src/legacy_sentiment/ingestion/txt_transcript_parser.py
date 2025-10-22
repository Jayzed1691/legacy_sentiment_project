#!/usr/bin/env python3

# txt_transcript_parser.py

from typing import List, Tuple
from transcript_structures import TranscriptData, Section, DialogueEntry

class TXTTranscriptParser:
	
	def parse(self, txt_data: str) -> TranscriptData:
		transcript_data = TranscriptData()
		lines = txt_data.splitlines()
		i = 0
		while i < len(lines):
			line = lines[i].strip()
			if line.startswith("Section:"):
				section_name = line.split("Section:")[1].strip()
				section = Section(name=section_name)
				transcript_data.sections.append(section)
				i = self._parse_section(lines, i + 1, section)
			else:
				i += 1
		return transcript_data
	
	def _parse_section(self, lines: List[str], start_index: int, section: Section) -> int:
		i = start_index
		current_subsection = None
		while i < len(lines):
			line = lines[i].strip()
			if line.startswith("Section:"):
				return i - 1
			elif line.startswith("Sub-Section:"):
				subsection_name = line.split("Sub-Section:")[1].strip()
				current_subsection = Section(name=subsection_name)
				section.subsections.append(current_subsection)
			elif ":" in line:
				try:
					speaker, dialogue = line.split(":", 1)
					speaker, role = self._extract_speaker_and_role(speaker)
					dialogue_entry = DialogueEntry(speaker=speaker.strip(), role=role, text=dialogue.strip())
					if current_subsection:
						current_subsection.dialogues.append(dialogue_entry)
					else:
						section.dialogues.append(dialogue_entry)
				except ValueError:
					if current_subsection and current_subsection.dialogues:
						current_subsection.dialogues[-1].text += " " + line
					elif section.dialogues:
						section.dialogues[-1].text += " " + line
			else:
				if current_subsection and current_subsection.dialogues:
					current_subsection.dialogues[-1].text += " " + line
				elif section.dialogues:
					section.dialogues[-1].text += " " + line
			i += 1
		return i
	
	def _extract_speaker_and_role(self, speaker: str) -> Tuple[str, str]:
		if "(" in speaker and ")" in speaker:
			name, role = speaker.split("(", 1)
			role = role.rstrip(")").strip()
			return name.strip(), role
		return speaker.strip(), ""