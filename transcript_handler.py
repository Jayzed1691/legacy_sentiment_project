#!/usr/bin/env python3

# transcript_handler.py

import json
from typing import Union, Dict, Any, Tuple, List
from transcript_structures import TranscriptData, Section, DialogueEntry
from json_transcript_parser import JSONTranscriptParser
from pdf_transcript_parser import PDFTranscriptParser
from txt_transcript_parser import TXTTranscriptParser

def parse_file(file_content: Any, file_type: str) -> TranscriptData:
	if file_type == 'json':
		json_data = json.loads(file_content)
		parser = JSONTranscriptParser()
		transcript_data = parser.parse(json_data)
	elif file_type == 'txt':
		parser = TXTTranscriptParser()
		transcript_data = parser.parse(file_content)
	elif file_type == 'pdf':
		parser = PDFTranscriptParser()
		transcript_data = parser.parse(file_content)
	else:
		raise ValueError(f"Unsupported file type: {file_type}")
		
	return transcript_data

def get_full_transcript(transcript_data: TranscriptData, indent_level: int = 0) -> str:
	transcript_str = ""
	for section in transcript_data.sections:
		transcript_str += format_section(section, indent_level)
	return transcript_str

def format_section(section: Section, indent_level: int) -> str:
	section_str = "  " * indent_level + f"Section: {section.name}\n"
	
	for dialogue in section.dialogues:
		section_str += "  " * (indent_level + 1) + f"{dialogue.speaker} ({dialogue.role}): {dialogue.text}\n"
		
	for subsection in section.subsections:
		section_str += format_section(subsection, indent_level + 1)
		
	return section_str

def process_transcript(file_path: str, file_type: str) -> Tuple[TranscriptData, str]:
	file_content = read_file(file_path, file_type)
	transcript_data = parse_file(file_content, file_type)
	formatted_transcript = get_full_transcript(transcript_data)
	return transcript_data, formatted_transcript

def read_file(file_path: str, file_type: str) -> Any:
	with open(file_path, 'r' if file_type in ['json', 'txt'] else 'rb') as file:
		return file.read()
	
def process_multiple_transcripts(file_paths: List[str], file_types: List[str]) -> List[Tuple[TranscriptData, str]]:
	results = []
	for file_path, file_type in zip(file_paths, file_types):
		try:
			result = process_transcript(file_path, file_type)
			results.append(result)
		except Exception as e:
			print(f"Error processing file {file_path}: {str(e)}")
	return results