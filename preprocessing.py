#!/usr/bin/env python3

# preprocessing.py

import re
from typing import List, Tuple, Union, Dict, Optional
import spacy
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import MWETokenizer, word_tokenize
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from concurrent.futures import ThreadPoolExecutor
import logging
import json
from file_utils import load_custom_stopwords, load_custom_entities, load_regex_patterns

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')

class Preprocessor:
	def __init__(self, config_path: str = "config.json", custom_stopwords=None, custom_entities=None):
		# Load configuration from file
		with open(config_path, 'r') as config_file:
			self.config = json.load(config_file)
			
		self.language = self.config.get('language', 'english')
		self.case_sensitive_stopwords = self.config.get('case_sensitive_stopwords', False)
		
		# Initialize stopwords
		self.stopwords_info = self._initialize_stopwords(custom_stopwords)
		
		# Load additional custom stopwords from config path if specified
		if "custom_stopwords_path" in self.config and not custom_stopwords:
			try:
				additional_stopwords = load_custom_stopwords(self.config["custom_stopwords_path"])
				self.stopwords_info = self._initialize_stopwords(additional_stopwords)
				logging.info("Additional custom stopwords loaded from config path.")
			except Exception as e:
				logging.error(f"Error loading custom stopwords from config path: {str(e)}")
				
		# Load custom named entities if provided
		self.custom_named_entities = custom_entities or {}
		if "custom_named_entities_path" in self.config and not custom_entities:
			self.load_custom_entities(self.config["custom_named_entities_path"])
		else:
			self.custom_named_entities = {}
			
		# Load custom regex patterns for extracting time periods if provided
		self.custom_regex_patterns = {}
		if "custom_regex_patterns_path" in self.config:
			try:
				self.custom_regex_patterns = load_regex_patterns(self.config["custom_regex_patterns_path"])
				logging.info("Custom regex patterns loaded.")
			except Exception as e:
				logging.error(f"Error loading custom regex patterns: {str(e)}")
				
		# Initialize MWETokenizer with common financial multi-word expressions
		self.mwe_tokenizer = None
		
		self.lemmatizer = WordNetLemmatizer()
		
	def _initialize_stopwords(self, custom_stopwords):
		stopwords_info = {}
		try:
			# Initialize stopwords if not already done
			if not hasattr(self, 'stop_words'):
				self.stop_words = set(stopwords.words(self.language))
				
			initial_count = len(self.stop_words)
			stopwords_info['initial'] = initial_count
			
			if custom_stopwords:
				if isinstance(custom_stopwords, dict) and "stopwords" in custom_stopwords:
					custom_stops = custom_stopwords["stopwords"]
				elif isinstance(custom_stopwords, list):
					custom_stops = custom_stopwords
				else:
					custom_stops = []
					logging.warning("Unexpected format for custom stopwords. Expected a dict with 'stopwords' key or a list.")
					
				if custom_stops:
					if self.case_sensitive_stopwords:
						self.stop_words.update(custom_stops)
					else:
						self.stop_words.update(word.lower() for word in custom_stops)
						
				stopwords_info['custom_added'] = len(self.stop_words) - initial_count
			else:
				stopwords_info['custom_added'] = 0
				
			stopwords_info['final'] = len(self.stop_words)
			
			return stopwords_info
		except Exception as e:
			logging.error(f"Error initializing stopwords: {str(e)}")
			return {}
		
	def get_stopwords_info(self):
		return self.stopwords_info
	
	def load_custom_entities(self, custom_entities_file: str):
		"""Load custom entities from a JSON file."""
		try:
			with open(custom_entities_file, 'r') as f:
				self.custom_named_entities = json.load(f)
			logging.info(f"Loaded custom entities from {custom_entities_file}")
		except Exception as e:
			logging.error(f"Error loading custom entities: {str(e)}")
			
	def _initialize_mwe_tokenizer(self):
		mwe_list = []
		for category, terms in self.custom_named_entities.items():
			for term in terms:
				if isinstance(term, dict):
					if category == "PERSON":
						full_name = term.get('full_name')
						if full_name:
							mwe_list.append(tuple(full_name.lower().split()))
						variations = term.get('variations', [])
						for variation in variations:
							mwe_list.append(tuple(variation.lower().split()))
					else:
						entity_term = term.get('term') or term.get('full_name')
						if entity_term:
							mwe_list.append(tuple(entity_term.lower().split()))
				elif isinstance(term, str):
					mwe_list.append(tuple(term.lower().split()))
				elif isinstance(term, list):
					mwe_list.append(tuple(word.lower() for word in term))
					
		logging.info(f"Initialized MWETokenizer with {len(mwe_list)} multi-word expressions")
		return MWETokenizer(mwe_list)
	
	def clean_text(self, text: str) -> str:
		"""Clean the input text by removing special characters and extra whitespace."""
		text = self._remove_html_tags(text)
		text = self._remove_punctuation(text)
		text = self._remove_extra_whitespace(text)
		text = self.lowercase_text(text)  # Use the new method here
		text = self.fix_specific_issues(text)  # Apply our new method
		return text.strip()
	
	def _remove_html_tags(self, text: str) -> str:
		"""Remove HTML tags from text."""
		return re.sub(r'<[^>]+>', ' ', text)
	
	def _remove_punctuation(self, text: str) -> str:
		"""
		Remove punctuation while retaining financial symbols ($, %), 
		hyphens, apostrophes, quotation marks, and in-number punctuation.
		"""
		keep_chars = set('$%\'-"')
		cleaned_chars = []
		i = 0
		while i < len(text):
			char = text[i]
			
			# Handle numbers with decimal points or commas
			if char.isdigit():
				num_str = char
				j = i + 1
				while j < len(text) and (text[j].isdigit() or text[j] in '.,'):
					if text[j] in '.,':
						if j+1 < len(text) and text[j+1].isdigit():
							num_str += text[j]
						else:
							break
					else:
						num_str += text[j]
					j += 1
				cleaned_chars.append(num_str)
				i = j
				continue
			
			# Handle percentages and apostrophes
			elif char in '%\'':
				if cleaned_chars and cleaned_chars[-1].isspace():
					cleaned_chars[-1] = char  # Replace the space with % or '
				else:
					cleaned_chars.append(char)
					
			# Handle other characters
			elif char.isalnum() or char.isspace() or char in keep_chars:
				cleaned_chars.append(char)
				
			# Replace other punctuation with space
			elif cleaned_chars and not cleaned_chars[-1].isspace():
				cleaned_chars.append(' ')
				
			i += 1
			
		return ''.join(cleaned_chars).strip()
	
	def _remove_extra_whitespace(self, text: str) -> str:
		"""Remove extra whitespace from text and handle special cases."""
		# First, handle special cases, but ONLY if not preceded by a digit or $
		text = re.sub(r'(?<![\d\$])\s+%', '%', text)  # Remove space before % if not after digit or $
		text = re.sub(r'(?<![\d\$])\s+\'', '\'', text) # Remove space before ' if not after digit or $
		text = re.sub(r'-\s+', '-', text)  # Remove space after hyphen
		text = re.sub(r'\s+-', '-', text)  # Remove space before hyphen
		
		# Then remove any remaining multiple spaces
		text = re.sub(r'\s+', ' ', text)
		
		return text.strip()
	
	def lowercase_text(self, text: str) -> str:
		"""
		Converts text to lowercase while preserving entities (conditional).
		"""
		words = word_tokenize(text)
		pos_tags = pos_tag(words)
		lowered_words = [word.lower() if tag not in ('NNP', 'NNPS') else word for word, tag in pos_tags]
		return ' '.join(lowered_words)
	
	def fix_specific_issues(self, text: str) -> str:
		"""
		Fix specific issues:
		1. Remove leading spaces before '%' signs
		2. Lowercase the first word of each sentence unless it's a proper noun
		"""
		# Remove leading spaces before '%' signs
		#text = re.sub(r'\s+%', '%', text)
		
		# Remove leading spaces before '%' signs and apostrophes
		text = re.sub(r'\s+([%\'])', r'\1', text)
		
		# Split the text into sentences
		sentences = re.split(r'(?<=[.!?])\s+', text)
		
		fixed_sentences = []
		for sentence in sentences:
			words = sentence.split()
			if words:
				# Check if the first word is not a proper noun (simple check)
				if not words[0].isupper() or words[0] in ['I', 'I\'m', 'I\'ve', 'I\'ll', 'I\'d']:
					words[0] = words[0].lower()
			fixed_sentences.append(' '.join(words))
			
		return ' '.join(fixed_sentences)
	
	def tokenize(self, text: Union[str, List[str]]) -> List[str]:
		"""Tokenize the input text using MWETokenizer and fall back to spaCy if needed."""
		# Initialize MWE tokenizer if not already done
		if self.mwe_tokenizer is None:
			self.mwe_tokenizer = self._initialize_mwe_tokenizer()
			
		# Handle input that's already a list of tokens
		if isinstance(text, list):
			return self.mwe_tokenizer.tokenize(text)
		
		# Handle string input
		if isinstance(text, str):
			# Convert text to lowercase for case-insensitive tokenization
			lower_text = text.lower()
			
			# First, tokenize with word_tokenize
			initial_tokens = word_tokenize(lower_text)
			
			# Then, apply MWETokenizer
			mwe_tokens = self.mwe_tokenizer.tokenize(initial_tokens)
			
			# If MWETokenizer produces unexpected results, fall back to spaCy
			if not mwe_tokens or any(len(token) > 50 for token in mwe_tokens):
				tokens = [token.text.lower() for token in nlp(lower_text)]
			else:
				tokens = mwe_tokens
				
			return self._post_process_tokens(tokens)
		
		# If input is neither a string nor a list, raise an error
		raise ValueError(f"Input must be either a string or a list of strings, not {type(text)}")
		
	def _post_process_tokens(self, tokens: List[str]) -> List[str]:
		"""Post-process tokens to handle contractions and other special cases."""
		processed_tokens = []
		i = 0
		while i < len(tokens):
			if tokens[i] == "'" and i > 0 and i < len(tokens) - 1:
				# Join contractions like "don't" or possessives
				processed_tokens[-1] += tokens[i] + tokens[i + 1]
				i += 2  # Skip the next token
			else:
				processed_tokens.append(tokens[i])
				i += 1
		return processed_tokens
	
	def remove_stopwords(self, tokens: List[str]) -> List[str]:
		"""
		Removes stopwords while preserving financial terms, named entities, and multi-word expressions.
		"""
		# Note: named entities and MWEs should already be preserved from earlier steps
		# We'll still check for financial terms
		
		# Remove stopwords
		if self.case_sensitive_stopwords:
			return [token for token in tokens if token not in self.stop_words or token in self.custom_named_entities]
		else:
			return [token for token in tokens if token.lower() not in self.stop_words or token in self.custom_named_entities]
		
	def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
		"""
		Lemmatize tokens while preserving all custom named entities and considering POS tags.
		"""
		lemmatized_tokens = []
		
		# Convert all custom named entities to lowercase for case-insensitive matching
		custom_entities = set(entity.lower() for entity in self.custom_named_entities)
		
		pos_tags = pos_tag(tokens)
		
		for token, pos in pos_tags:
			if token.lower() in custom_entities:
				# Preserve custom named entities
				lemmatized_tokens.append(token)
			else:
				# Map POS tag to WordNet POS
				wordnet_pos = self._get_wordnet_pos(pos)
				if wordnet_pos:
					lemmatized_tokens.append(self.lemmatizer.lemmatize(token, pos=wordnet_pos))
				else:
					lemmatized_tokens.append(self.lemmatizer.lemmatize(token))
					
		return lemmatized_tokens
	
	def _get_wordnet_pos(self, treebank_tag: str) -> Optional[str]:
		"""
		Map POS tag to first character used by WordNet lemmatizer.
		"""
		tag = treebank_tag[0].upper()
		tag_dict = {"J": wordnet.ADJ,
					"N": wordnet.NOUN,
					"V": wordnet.VERB,
					"R": wordnet.ADV}
		return tag_dict.get(tag, wordnet.NOUN)
	
	def extract_named_entities(self, text: str) -> List[Tuple[str, str]]:
		"""Extract named entities from text, prioritizing custom entities and longer matches."""
		entity_spans = {}
		
		# First, identify custom entities
		for category, terms in self.custom_named_entities.items():
			for term in terms:
				if isinstance(term, dict):
					if category == "PERSON":
						full_name = term.get('full_name')
						if full_name:
							self._add_entity_to_spans(text, full_name, category, entity_spans)
						variations = term.get('variations', [])
						for variation in variations:
							# Only add variations if they don't overlap with the full name
							if not self._overlaps_with_full_name(text, variation, full_name):
								self._add_entity_to_spans(text, variation, category, entity_spans)
					else:
						entity_term = term.get('term') or term.get('full_name')
						if entity_term:
							self._add_entity_to_spans(text, entity_term, category, entity_spans)
				elif isinstance(term, str):
					self._add_entity_to_spans(text, term, category, entity_spans)
					
		# Then, use NLTK's ne_chunk for additional entities
		tokens = word_tokenize(text)
		pos_tags = pos_tag(tokens)
		chunked = ne_chunk(pos_tags)
		for subtree in chunked:
			if isinstance(subtree, Tree):
				entity_text = " ".join([word for word, tag in subtree.leaves()])
				entity_type = subtree.label()
				self._add_entity_to_spans(text, entity_text, entity_type, entity_spans)
				
		# Then, use spaCy for additional entities, but only if they don't overlap with custom entities
		doc = nlp(text)
		for ent in doc.ents:
			if not any(start <= ent.start_char < ent.end_char <= end for (start, end) in entity_spans):
				self._add_entity_to_spans(text, ent.text, ent.label_, entity_spans)
				
		# Remove overlapping entities, prioritizing custom entities and longer spans
		non_overlapping_spans = self._remove_overlapping_spans(entity_spans)
		
		# Convert spans back to a list of tuples
		entities = [(text[start:end], label) for (start, end), label in sorted(non_overlapping_spans.items()) if start < end]
		
		return entities
	
	def _overlaps_with_full_name(self, text: str, variation: str, full_name: str) -> bool:
		"""Check if a name variation overlaps with the full name in the text."""
		variation_start = text.lower().find(variation.lower())
		full_name_start = text.lower().find(full_name.lower())
		
		if variation_start == -1 or full_name_start == -1:
			return False
		
		variation_end = variation_start + len(variation)
		full_name_end = full_name_start + len(full_name)
		
		return (variation_start <= full_name_start < variation_end) or \
			(variation_start < full_name_end <= variation_end) or \
			(full_name_start <= variation_start < full_name_end)
	
	def _add_entity_to_spans(self, text: str, entity: str, category: str, entity_spans: Dict[Tuple[int, int], str]):
		"""Helper method to add an entity to the spans dictionary."""
		start = 0
		while True:
			start = text.lower().find(entity.lower(), start)
			if start == -1:  # No more occurrences
				break
			end = start + len(entity)
			# Check for word boundaries
			if (start == 0 or not text[start-1].isalnum()) and (end == len(text) or not text[end].isalnum()):
				entity_spans[(start, end)] = category
			start = end  # Move to the end of the current match to find the next one
		
	def _remove_overlapping_spans(self, spans: Dict[Tuple[int, int], str]) -> Dict[Tuple[int, int], str]:
		"""Remove overlapping spans, prioritizing custom entities and longer spans."""
		sorted_spans = sorted(spans.items(), key=lambda x: (x[0][0], -x[0][1], self._get_entity_priority(x[1])))
		non_overlapping = {}
		last_end = -1
		
		for (start, end), label in sorted_spans:
			if start >= last_end:
				non_overlapping[(start, end)] = label
				last_end = end
				
		return non_overlapping
	
	def _get_entity_priority(self, entity_type: str) -> int:
		"""Return a priority value for entity types. Lower values have higher priority."""
		custom_priorities = {
			'PERSON': 0,
			'FINANCIAL_VOCABULARY': 1,
			'ORGANIZATION': 2,
			'ORG': 3,
			'TITLE': 4,
			'PRODUCT': 5,
			'QUANTITY': 6,
			'MONEY': 7,
			'DATE': 8,
			'TIME': 9,
			'GPE': 10,
			'LOCATION': 11,  # NLTK entity type
			'LOC': 12,  # NLTK entity type (Geo-Political Entity)
			'EVENT': 13,
			# Add other custom entity types here
		}
		return custom_priorities.get(entity_type, 100)  # Default priority for unknown types
	
	def extract_time_periods(self, text: str) -> List[str]:
		"""Extract time periods from text using custom regex patterns."""
		time_periods = []
		if "time_periods" in self.custom_regex_patterns:
			for pattern in self.custom_regex_patterns["time_periods"]:
				matches = re.findall(pattern, text)
				time_periods.extend(matches)
		return time_periods
	
	def preprocess(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
		"""
		Preprocess the input text based on the configuration.
	
		Args:
			text (Union[str, List[str]]): The text to be preprocessed.
	
		Returns:
			Union[str, List[str]]: The preprocessed text or tokens.
		"""
		try:
			logging.debug(f"Preprocess input type: {type(text)}")
			logging.debug(f"Preprocess input content: {text[:100] if isinstance(text, str) else text[:10]}")
			
			# If input is a list, join it into a string
			if isinstance(text, list):
				text = ' '.join(text)
				
			if self.config.get("clean", True):
				text = self.clean_text(text)
				
			# Extract named entities
			named_entities = self.extract_named_entities(text) if self.config.get("extract_entities", False) else []
			
			# Apply MWE tokenizer
			mwe_tokens = self.apply_mwe_tokenizer(text)
			
			# Combine named entities and MWE tokens
			preserved_tokens = [entity[0] for entity in named_entities] + mwe_tokens
			
			# Tokenize remaining text
			remaining_text = self.remove_preserved_tokens(text, preserved_tokens)
			tokens = self.tokenize(remaining_text)
			
			# Combine all tokens
			all_tokens = preserved_tokens + tokens
			
			if self.config.get("lemmatize", True):
				all_tokens = self.lemmatize_tokens(all_tokens)
				
			if self.config.get("remove_stopwords", True):
				all_tokens = self.remove_stopwords(all_tokens)
				
			if self.config.get("extract_time_periods", False):
				time_periods = self.extract_time_periods(' '.join(all_tokens))
				all_tokens.extend(time_periods)
				
			return all_tokens if self.config.get("tokenize", True) else ' '.join(all_tokens)
		
		except Exception as e:
			logging.error(f"Error during preprocessing: {e}")
			return text
		
	def apply_mwe_tokenizer(self, text: str) -> List[str]:
		"""Apply the Multi-Word Expression tokenizer to the text."""
		if self.mwe_tokenizer is None:
			self.mwe_tokenizer = self._initialize_mwe_tokenizer()
			
		# First, tokenize the text
		tokens = word_tokenize(text)
		
		# Then apply the MWE tokenizer
		mwe_tokens = self.mwe_tokenizer.tokenize(tokens)
		
		# Return only the multi-word expressions
		return [token for token in mwe_tokens if '_' in token]
	
	def remove_preserved_tokens(self, text: str, preserved_tokens: List[str]) -> str:
		"""Remove preserved tokens (named entities and MWEs) from the text."""
		for token in sorted(preserved_tokens, key=len, reverse=True):
			text = text.replace(token, '')
		return ' '.join(text.split())  # Remove extra whitespace
	
	def preprocess_parallel(self, texts: List[str]) -> List[Union[str, List[str]]]:
		"""Preprocess multiple texts in parallel."""
		with ThreadPoolExecutor() as executor:
			return list(executor.map(self.preprocess, texts))
		
if __name__ == "__main__":
	# Example usage
	preprocessor = Preprocessor(config_path="config.json")
	sample_texts = [
		"<p>Hello, world!</p> This is an example text.",
		"This text contains <b>HTML</b> tags and some punctuation!",
		"Let's see if it works well."
	]
	
	# Process texts in parallel
	processed_texts = preprocessor.preprocess_parallel(sample_texts)
	for original, processed in zip(sample_texts, processed_texts):
		logging.info(f"Original: {original}\nProcessed: {processed}\n")