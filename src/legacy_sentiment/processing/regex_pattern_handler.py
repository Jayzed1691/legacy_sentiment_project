#!/usr/bin/env python3

# regex_pattern_handler.py

import json
import logging
import re
from typing import List, Tuple, Dict, Union
import nltk
from nltk.tokenize import sent_tokenize
from custom_file_utils import load_regex_patterns

class RegexPatternHandler:    
	def __init__(self, regex_patterns_files: Union[str, List[str]]):
		self.regex_patterns = load_regex_patterns(regex_patterns_files)
		
	def _extract_custom_patterns(self, text: str) -> List[Tuple[str, str, str, int, int]]:
		all_matches = []
		
		# Define priority order for categories
		category_priority = ['financial_patterns', 'time_patterns', 'currency_patterns']
		
		for category in category_priority:
			if category in self.regex_patterns:
				for pattern_dict in self.regex_patterns[category]:
					pattern = pattern_dict['pattern']
					label = pattern_dict.get('label', category)
					for match in re.finditer(pattern, text, re.IGNORECASE):
						all_matches.append((match.group(), label, 'Regex', match.start(), match.end()))
						
		# Sort matches by start position and then by length (longer matches first)
		all_matches.sort(key=lambda x: (x[3], -len(x[0])))
		
		# Remove overlapping matches, keeping the first (longest) match
		non_overlapping = []
		last_end = -1
		for match in all_matches:
			if match[3] >= last_end:
				non_overlapping.append(match)
				last_end = match[4]
				
		return non_overlapping
	
	def process_text(self, text: str) -> List[Tuple[str, str, str, int, int]]:
		return self._extract_custom_patterns(text)