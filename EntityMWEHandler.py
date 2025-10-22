#!/usr/bin/env python3

# EntityMWEHandler.py

from __future__ import annotations
import logging
import nltk
import json
from typing import List, Tuple, Dict, Any, Union, Optional
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from custom_file_utils import (
	load_custom_entities, 
	load_multi_word_entries, 
	load_regex_patterns,
	load_json_file,
	load_language_data,
	load_custom_stopwords
)
from custom_entity_handler import CustomEntityHandler
from mwe_handler import MWEHandler
from regex_pattern_handler import RegexPatternHandler
from spacy_handler import SpaCyHandler
from stopword_handler import StopwordHandler
from text_cleaner import TextCleaner
from token_processor import TokenProcessor

# Download required NLTK data
nltk.download('punkt', quiet=True)

# Type aliases for clarity
EntityToken = Tuple[str, str, str, int, int]  # (text, label, source, start, end)
ProcessedToken = Tuple[str, str, str, str, str]  # (text, lemma, pos, action, source)
SentenceResult = Tuple[List[EntityToken], List[ProcessedToken], str]

class EntityMWEHandler:
	"""
	Handles entity extraction, multi-word expression identification, and text processing.
	
	Coordinates between various processors to identify entities, protect them during
	processing, and clean the text while preserving important linguistic elements.
	"""
	
	def __init__(self, 
				custom_entities_files: Optional[Union[str, List[str]]] = None, 
				multi_word_entries_files: Optional[Union[str, List[str]]] = None, 
				regex_patterns_files: Optional[Union[str, List[str]]] = None, 
				language_data: Optional[Dict[str, Any]] = None,
				spacy_handler: Optional[SpaCyHandler] = None,
				custom_stopwords_files: Optional[Union[str, List[str]]] = None,
				preprocessing_config: Optional[Dict[str, Any]] = None) -> None:
		"""
		Initialize the EntityMWEHandler with all necessary components.
		
		Args:
			custom_entities_files: Optional path(s) to custom entity definition files
			multi_word_entries_files: Optional path(s) to multi-word expression files
			regex_patterns_files: Optional path(s) to regex pattern files
			language_data: Optional preprocessed language data dictionary
			spacy_handler: Optional initialized SpaCy handler instance
			custom_stopwords_files: Optional path(s) to custom stopword files
			preprocessing_config: Optional configuration dictionary for preprocessing options
		"""
		self.config = preprocessing_config or {}
		
		# Load shared data with fallbacks for missing files
		self.custom_entities = load_custom_entities(custom_entities_files) if custom_entities_files else {}
		self.mwe_entries = load_multi_word_entries(multi_word_entries_files) if multi_word_entries_files else {}
		
		# Initialize handlers with optional components
		self.custom_entity_handler = (CustomEntityHandler(custom_entities_files) 
							if custom_entities_files else None)
		self.mwe_handler = (MWEHandler(multi_word_entries_files) 
							if multi_word_entries_files else None)
		self.regex_handler = (RegexPatternHandler(regex_patterns_files) 
							if regex_patterns_files else None)
		
		self.nlp = spacy.load("en_core_web_sm")
		self.language_data = language_data or {}
		self.spacy_handler = spacy_handler or SpaCyHandler({})
		
		# Initialize processors with optional components
		self.stopword_handler = StopwordHandler(
			custom_stopwords_files if custom_stopwords_files else [],
			case_sensitive=self.config.get('case_sensitive_stopwords', False)
		)
		self.token_processor = TokenProcessor(self.stopword_handler)
		
		# Use already loaded data for TextCleaner
		self.text_cleaner = TextCleaner(
			self.custom_entities, 
			self.mwe_entries,
			config=self.config  # Add this line
		)
		
	def process_text(self, text: str) -> List[Tuple[str, SentenceResult]]:
		"""
		Process text with entity extraction, token processing, and cleaning.
		
		Args:
			text: Input text to process
			
		Returns:
			List of tuples containing:
				- Original sentence with number
				- Tuple of (identified_tokens, processed_tokens, cleaned_text)
		"""
		if not self.config.get('tokenize', True):
			return [(text, ([], [], text))]
		
		sentences = sent_tokenize(text)
		results = []
		
		current_position = 0
		for sentence_num, sentence in enumerate(sentences, 1):
			processed_sentence = self.process_sentence(sentence, current_position)
			results.append((f"Sentence {sentence_num}: {sentence}", processed_sentence))
			current_position += len(sentence) + 1
			
		return results
	
	def process_sentence(self, sentence: str, offset: int = 0) -> SentenceResult:
		"""
		Process a single sentence with available components.
		
		Args:
			sentence: The sentence to process
			offset: Character offset for the sentence in the original text
			
		Returns:
			Tuple containing:
				- List of identified entity tokens
				- List of processed tokens
				- Cleaned text
		"""
		# First identify entities if enabled
		identified_tokens = []
		if self.config.get('extract_entities', True):
			identified_tokens = self.tokenize_and_identify(sentence, offset)
			
		# Process tokens and handle stopwords
		processed_tokens = self.token_processor.process_text(sentence, identified_tokens)
		
		# Clean text using configured options
		cleaned_text = self.text_cleaner.clean_text(
			text=sentence,
			entities=identified_tokens,
			processed_tokens=processed_tokens,
			use_lemma=self.config.get('lemmatize', True)
		)
		
		return identified_tokens, processed_tokens, cleaned_text
	
	def tokenize_and_identify(self, text: str, offset: int = 0) -> List[EntityToken]:
		"""
		Identify all entities in text using available handlers.
		
		Args:
			text: Text to process
			offset: Character offset in original text
			
		Returns:
			List of identified entity tokens with adjusted positions
		"""
		all_matches = []
		
		# Only use handlers if they were initialized
		if self.custom_entity_handler:
			all_matches.extend(self.custom_entity_handler.extract_named_entities(text))
			
		if self.mwe_handler:
			all_matches.extend(self.mwe_handler.extract_multi_word_expressions(text))
			
		if self.regex_handler:
			all_matches.extend(self.regex_handler.process_text(text))
			
		# SpaCy processing is always available
		doc = self.nlp(text)
		spacy_entities = [
			(ent.text, ent.label_, 'spaCy', ent.start_char, ent.end_char)
			for ent in doc.ents if not self.spacy_handler._contains_and(ent.text)
		]
		spacy_entities = self.spacy_handler.refine_entities(text, spacy_entities)
		all_matches.extend(spacy_entities)
		
		non_overlapping = self._remove_overlaps(all_matches)
		
		return [
			(entity, category, source, start + offset, end + offset)
			for entity, category, source, start, end in non_overlapping
		]
	
	def get_active_components(self) -> Dict[str, bool]:
		"""
		Get status of optional components.
		
		Returns:
			Dictionary indicating which components are active
		"""
		return {
			'custom_entities': bool(self.custom_entity_handler),
			'multi_word_expressions': bool(self.mwe_handler),
			'regex_patterns': bool(self.regex_handler),
			'language_data': bool(self.language_data),
			'custom_stopwords': bool(self.stopword_handler.custom_stopwords)
		}
	
	def _remove_overlaps(self, matches: List[EntityToken]) -> List[EntityToken]:
		"""
		Remove overlapping matches, prioritizing by source and length.
		
		Args:
			matches: List of identified entity tokens
			
		Returns:
			List of non-overlapping entity tokens
		"""
		sorted_matches = sorted(matches, key=lambda x: (x[3], -len(x[0]), self._get_source_priority(x[2])))
		non_overlapping = []
		for match in sorted_matches:
			if not any(self._is_overlapping(match, existing) for existing in non_overlapping):
				non_overlapping.append(match)
		return non_overlapping
	
	def _is_overlapping(self, match1: EntityToken, match2: EntityToken) -> bool:
		"""
		Check if two matches overlap in the text.
		
		Args:
			match1: First entity token
			match2: Second entity token
			
		Returns:
			True if the entities overlap, False otherwise
		"""
		_, _, _, start1, end1 = match1
		_, _, _, start2, end2 = match2
		return (start1 < end2 and start2 < end1) and match1 != match2
	
	def _is_protected(self, token: Any, identified_tokens: List[EntityToken]) -> bool:
		"""
		Check if a token is within a protected entity span.
		
		Args:
			token: spaCy token to check
			identified_tokens: List of identified entity tokens
			
		Returns:
			True if token is within a protected span, False otherwise
		"""
		token_start = token.idx
		token_end = token.idx + len(token.text)
		return any(
			(start <= token_start < end) or (start < token_end <= end)
			for _, _, _, start, end in identified_tokens
		)
	
	def _get_source_priority(self, source: str) -> int:
		"""
		Get priority for entity source type.
		
		Args:
			source: Source identifier string
			
		Returns:
			Priority value (lower is higher priority)
		"""
		priorities = {'Custom Entity': 1, 'MWE': 2, 'Regex': 0, 'spaCy': 3}
		return priorities.get(source, 4)  # Default priority for unknown sources