#!/usr/bin/env python3

# text_cleaner.py

import re
import spacy
from typing import Dict, List, Any, Tuple

class TextCleaner:
	def __init__(self, custom_named_entities: Dict[str, List[Dict[str, Any]]], 
				mwe_entries: Dict[str, List[Dict[str, Any]]],
				config: Dict[str, Any] = None):
		self.custom_entities = self._extract_entities(custom_named_entities)
		self.exceptions = self._extract_exceptions(mwe_entries)
		self.config = config or {}
		# Define punctuation to preserve in specific contexts
		self.preserve_punct = {'.', '!', '?', '%'}  # Only essential punctuation
		
	def clean_text(self, text: str, entities: List[Tuple[str, str, str, int, int]], 
			processed_tokens: List[Tuple[str, str, str, str, str]], 
			use_lemma: bool = True) -> str:
		# Create protected spans from entities
		protected_spans = set()
		entity_map = {}  # Map positions to complete entity text
		for entity_text, _, _, start, end in entities:
			protected_spans.update(range(start, end))
			entity_map[(start, end)] = entity_text
			
		# Process tokens
		cleaned_words = []
		current_pos = 0
		last_token_was_punct = False
		
		for token, lemma, pos, action, source in processed_tokens:
			token_start = text[current_pos:].find(token)
			if token_start == -1:
				continue
			token_start += current_pos
			token_end = token_start + len(token)
			
			is_protected = any(token_start + i in protected_spans for i in range(len(token)))
			
			if is_protected:
				entity = next(
					(text for (start, end), text in entity_map.items() 
					if start <= token_start < end),
					None
				)
				if entity and (not cleaned_words or cleaned_words[-1] != entity):
					cleaned_words.append(entity)
					last_token_was_punct = False
			elif action == 'Keep':
				if self._is_punctuation(token):
					# Only keep essential punctuation
					if token in self.preserve_punct:
						if cleaned_words and not last_token_was_punct:
							cleaned_words[-1] = cleaned_words[-1].rstrip()
							cleaned_words.append(token)
							last_token_was_punct = True
				else:
					word = lemma if use_lemma else token
					cleaned_words.append(word.lower())
					last_token_was_punct = False
					
			current_pos = token_end
			
		if not cleaned_words:
			return ""
		
		# Join words with appropriate spacing
		cleaned_text = cleaned_words[0]
		for word in cleaned_words[1:]:
			if not self._is_punctuation(word):
				cleaned_text += " "
			cleaned_text += word
			
		return cleaned_text.strip()
	
	def _is_punctuation(self, token: str) -> bool:
		"""Check if token consists entirely of punctuation characters."""
		return all(not c.isalnum() for c in token)
	
	# [Previous _extract_entities and _extract_exceptions methods remain unchanged]