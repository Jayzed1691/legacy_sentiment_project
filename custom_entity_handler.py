#!/usr/bin/env python3

# custom_entity_handler.py

import json
import logging
from typing import List, Tuple, Dict, Any, Union
from custom_file_utils import load_custom_entities

class CustomEntityHandler:
	def __init__(self, custom_entities_files: Union[str, List[str]]):
		self.custom_named_entities = load_custom_entities(custom_entities_files)
		
	def extract_named_entities(self, text: str) -> List[Tuple[str, str, str, int, int]]:
		all_matches = []
		
		for category, terms in self.custom_named_entities.items():
			for term in terms:
				if isinstance(term, dict):
					entity_term = term.get('term') or term.get('full_name')
					if entity_term:
						self._add_entity_to_spans(text, entity_term, category, all_matches)
					for variation in term.get('variations', []):
						if not self._overlaps_with_full_name(text, variation, entity_term):
							self._add_entity_to_spans(text, variation, category, all_matches)
				elif isinstance(term, str):
					self._add_entity_to_spans(text, term, category, all_matches)
					
		# Sort matches by start position and then by length (longer matches first)
		all_matches.sort(key=lambda x: (x[3], -len(x[0])))
		
		final_entities = []
		for entity, category, source, start, end in all_matches:
			if not any(self._is_complete_overlap(start, end, other_start, other_end) 
						for _, _, _, other_start, other_end in final_entities):
				final_entities.append((entity, category, source, start, end))
			
		return final_entities
	
	def _is_complete_overlap(self, start1: int, end1: int, start2: int, end2: int) -> bool:
		return start1 >= start2 and end1 <= end2 and (start1, end1) != (start2, end2)
	
	def _add_entity_to_spans(self, text: str, entity: str, category: str, matches: List[Tuple[str, int, int, str, str]]):
		start = 0
		while True:
			start = text.lower().find(entity.lower(), start)
			if start == -1:  # No more occurrences
				break
			end = start + len(entity)
			# Check for word boundaries
			if (start == 0 or not text[start-1].isalnum()) and (end == len(text) or not text[end].isalnum()):
				matches.append((text[start:end], category, 'Custom Entity', start, end))
			start = end  # Move to the end of the current match to find the next one
		
	def _overlaps_with_full_name(self, text: str, variation: str, full_name: str) -> bool:
		variation_start = text.lower().find(variation.lower())
		full_name_start = text.lower().find(full_name.lower())
		
		if variation_start == -1 or full_name_start == -1:
			return False
		
		variation_end = variation_start + len(variation)
		full_name_end = full_name_start + len(full_name)
		
		return (variation_start <= full_name_start < variation_end) or \
				(variation_start < full_name_end <= variation_end) or \
				(full_name_start <= variation_start < full_name_end)