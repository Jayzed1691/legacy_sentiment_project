#!/usr/bin/env python3

# pdf_transcript_parser.py

import pdfplumber
import re
from typing import List, Optional, Tuple

from legacy_sentiment.data_models.transcript_structures import DialogueEntry, Section, TranscriptData

class PDFTranscriptParser:
	def __init__(self):
		pass
		
	def extract_text_from_pdf(self, uploaded_file):
		full_text = ""
		with pdfplumber.open(uploaded_file) as pdf:
			for page in pdf.pages:
				full_text += page.extract_text() if page.extract_text() else ""
		return full_text
	
	def clean_extracted_text(self, raw_text):
		# First, protect the content within angle brackets (speaker names and roles)
		protected_text = re.sub(r'(<[^>]+>)', lambda m: m.group(1).replace(' ', '|SPACE|'), raw_text)
		
		# Now apply the cleaning operations
		cleaned_text = protected_text.replace("-\n", "").replace("\n", " ")
		cleaned_text = re.sub(r'(<[^>]+:.*?>)', r'\n\1', cleaned_text)
		cleaned_text = re.sub(r'(\[SECTION: .+?\])', r'\n\1\n', cleaned_text)
		cleaned_text = re.sub(r'(\[QUESTION \d+\])', r'\n\1\n', cleaned_text)
		cleaned_text = re.sub(r'(\[ANSWER \d+\])', r'\n\1\n', cleaned_text)
		
		# Restore the spaces in speaker names and roles
		cleaned_text = cleaned_text.replace('|SPACE|', ' ')
		
		return cleaned_text
	
	def parse_pdf_text(self, cleaned_text):
		sections_raw = re.split(r'(\[SECTION: .+?\])', cleaned_text)
		transcript_data = TranscriptData(sections=[])
		
		current_section = None
		current_subsection = None
		
		for section_raw in sections_raw:
			# Detect new sections
			section_match = re.match(r'\[SECTION: (.+?)\]', section_raw.strip())
			if section_match:
				section_name = section_match.group(1)
				current_section = Section(name=section_name, dialogues=[], subsections=[])
				transcript_data.sections.append(current_section)
				continue
			
			# Handle dialogues and Q&A within the section
			lines = section_raw.splitlines()
			for line in lines:
				line = line.strip()
				
				# Detect general dialogue
				dialogue_match = re.match(r'<([^:]+):([^>]+)> (.+)', line)
				if dialogue_match and current_section:
					speaker = dialogue_match.group(1).strip()
					role = dialogue_match.group(2).strip()
					text = dialogue_match.group(3).strip()
					dialogue_entry = DialogueEntry(speaker=speaker, role=role, text=text)
					if current_subsection:
						current_subsection.dialogues.append(dialogue_entry)
					else:
						current_section.dialogues.append(dialogue_entry)
					continue
				
				# Detect start of a question or answer
				subsection_match = re.match(r'\[(QUESTION|ANSWER) (\d+)\]', line)
				if subsection_match:
					subsection_type = subsection_match.group(1)
					subsection_number = subsection_match.group(2)
					current_subsection = Section(name=f"{subsection_type} {subsection_number}", dialogues=[])
					current_section.subsections.append(current_subsection)
					continue
				
				# Add text to current subsection if it exists
				if current_subsection and line:
					current_subsection.dialogues.append(DialogueEntry(speaker="", role="", text=line))
					
		return transcript_data
	
	def parse(self, uploaded_file):
		try:
			# Handle uploaded_file as a string path if provided directly
			if isinstance(uploaded_file, str):
				raw_text = self.extract_text_from_pdf(uploaded_file)
			else:
				raw_text = self.extract_text_from_pdf(uploaded_file)
				
			cleaned_text = self.clean_extracted_text(raw_text)
			return self.parse_pdf_text(cleaned_text)
		except Exception as e:
			print(f"Error processing file: {str(e)}")
			return None