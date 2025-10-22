#!/usr/bin/env python3

# stopword_handler.py

import json
from typing import List, Tuple

import spacy

from legacy_sentiment.utils.custom_file_utils import load_custom_stopwords

class StopwordHandler:
	def __init__(self, custom_stopwords_files: List[str], case_sensitive: bool = False):
		self.nlp = spacy.load("en_core_web_sm")
		self.case_sensitive = case_sensitive
		self.custom_stopwords = self.load_custom_stopwords(custom_stopwords_files)
		self.all_stopwords = self.nlp.Defaults.stop_words.union(self.custom_stopwords)
		
	def load_custom_stopwords(self, custom_stopwords_files: List[str]) -> set:
		stopwords_dict = load_custom_stopwords(custom_stopwords_files)
		stopwords = set(stopwords_dict.get("stopwords", []))
		if not self.case_sensitive:
			stopwords = {word.lower() for word in stopwords}
		return stopwords
	
	def process_text_from_doc(self, doc: spacy.tokens.Doc) -> List[Tuple[str, str, str, str, str]]:
		"""Process tokens from pre-processed Doc."""
		processed_tokens = []
		
		# Create case-sensitive sets once for the whole doc
		if not self.case_sensitive:
			custom_stops = {word.lower() for word in self.custom_stopwords}
			standard_stops = {word.lower() for word in self.nlp.Defaults.stop_words}
		else:
			custom_stops = self.custom_stopwords
			standard_stops = self.nlp.Defaults.stop_words
			
		for token in doc:
			token_text = token.text if self.case_sensitive else token.text.lower()
			token_lemma = token.lemma_ if self.case_sensitive else token.lemma_.lower()
			
			# Check stopword status using pre-computed sets
			is_custom_stop = token_text in custom_stops or token_lemma in custom_stops
			is_standard_stop = token_text in standard_stops or token_lemma in standard_stops
			
			processed_tokens.append((
				token.text,
				token.lemma_,
				token.pos_,
				'Stopword' if (is_custom_stop or is_standard_stop) else 'Keep',
				'Custom' if is_custom_stop 
				else 'Standard' if is_standard_stop 
				else 'N/A'
			))
			
		return processed_tokens