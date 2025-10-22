#!/usr/bin/env python3

# mwe_handler.py

import json
import logging
from typing import List, Tuple, Dict, Any, Union
from legacy_sentiment.utils.custom_file_utils import load_multi_word_entries

class MWEHandler:
	def __init__(self, multi_word_entries_files: Union[str, List[str]]):
		self.multi_word_entries = load_multi_word_entries(multi_word_entries_files)
		
	def extract_multi_word_expressions(self, text: str) -> List[Tuple[str, str, str, int, int]]:
		all_matches = []
		
		for category, terms in self.multi_word_entries.items():
			for term in terms:
				if isinstance(term, dict) and 'term' in term:
					self._add_mwe_to_spans(text, term['term'], category, all_matches)
				elif isinstance(term, str):
					self._add_mwe_to_spans(text, term, category, all_matches)
					
		# Sort matches by start position and then by length (longer matches first)
		all_matches.sort(key=lambda x: (x[3], -len(x[0])))
		
		final_mwes = []
		for mwe, category, source, start, end in all_matches:
			if not any(self._is_complete_overlap(start, end, other_start, other_end) 
						for _, _, _, other_start, other_end in final_mwes):
				final_mwes.append((mwe, category, source, start, end))
			
		return final_mwes
	
	def _is_complete_overlap(self, start1: int, end1: int, start2: int, end2: int) -> bool:
		return start1 >= start2 and end1 <= end2 and (start1, end1) != (start2, end2)
	
	def _add_mwe_to_spans(self, text: str, mwe: str, category: str, matches: List[Tuple[str, int, int, str, str]]):
		start = 0
		while True:
			start = text.lower().find(mwe.lower(), start)
			if start == -1:  # No more occurrences
				break
			end = start + len(mwe)
			# Check for word boundaries
			if (start == 0 or not text[start-1].isalnum()) and (end == len(text) or not text[end].isalnum()):
				matches.append((text[start:end], category, 'MWE', start, end))
			start = end  # Move to the end of the current match to find the next one