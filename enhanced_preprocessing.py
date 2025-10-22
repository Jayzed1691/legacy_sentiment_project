#!/usr/bin/env python3

# enhanced_preprocessing.py

import re
import nltk
from nltk.tokenize import MWETokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
from typing import List, Union
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

class TextPreprocessor:
	def __init__(self, language='english', custom_stopwords=None, regex_patterns=None, custom_entities=None, config=None):
		self.language = language
		self.config = config if config else {}
		# Load NLTK stopwords
		self.stopwords = set(stopwords.words(language))
		# Load custom stopwords from provided data
		if custom_stopwords:
			self.stopwords.update(custom_stopwords)
			
		# Load regex patterns (e.g., for dates, financial terms, and currencies) from provided data
		self.regex_patterns = regex_patterns if regex_patterns else {}
		
		# Load custom named entities and financial vocabulary from provided data
		self.custom_named_entities = []
		self.financial_terms = set()
		if custom_entities:
			# Extract named entities from all relevant categories
			for category in ['COMPANY', 'PERSON', 'ORG', 'PRODUCT', 'LOCATION', 'CURRENCY', 'HOLIDAY', 'TIME_PERIOD']:
				entities = custom_entities.get(category, [])
				for entity in entities:
					self.custom_named_entities.append(entity['term'])
			# Extract financial vocabulary
			financial_vocab = custom_entities.get('FINANCIAL_VOCABULARY', [])
			for term in financial_vocab:
				self.financial_terms.add(term['term'].lower())
				
		self.lemmatizer = WordNetLemmatizer()
		self.mwe_tokenizer = MWETokenizer([('stock', 'market'), ('gross', 'merchandise', 'value'), ('new', 'york')])
		
	def lowercase_text(self, text: str) -> str:
		"""
		Converts text to lowercase while preserving entities (conditional).
		"""
		# Implementing a simple rule to avoid lowercasing proper nouns (entities will be properly handled in Step 2)
		words = word_tokenize(text)
		pos_tags = pos_tag(words)
		lowered_words = [word.lower() if tag not in ('NNP', 'NNPS') else word for word, tag in pos_tags]
		return ' '.join(lowered_words)
	
	def remove_punctuation(self, text: str) -> str:
		"""
		Removes punctuation while retaining financial symbols.
		"""
		return re.sub(r'(?<!\$)(?<!%)\W+', ' ', text)
	
	def tokenize_text(self, text: str) -> List[str]:
		"""
		Tokenizes text, preserving multi-word expressions.
		"""
		return self.mwe_tokenizer.tokenize(word_tokenize(text))
	
	def remove_stopwords(self, tokens: List[str]) -> List[str]:
		"""
		Removes stopwords except for essential financial terms.
		"""
		return [token for token in tokens if token.lower() not in self.stopwords or token.lower() in self.financial_terms]
	
	def identify_named_entities(self, tokens: List[str]) -> List[str]:
		"""
		Identifies named entities in the text and returns a list of entity tokens.
		"""
		named_entities = self.custom_named_entities[:]
		chunked = ne_chunk(pos_tag(tokens))
		for subtree in chunked:
			if isinstance(subtree, Tree) and subtree.label() in ('PERSON', 'ORGANIZATION', 'GPE', 'LOCATION'):
				named_entity = " ".join([token for token, pos in subtree.leaves()])
				named_entities.append(named_entity)
		return named_entities
	
	def conditional_lemmatization(self, tokens: List[str], named_entities: List[str]) -> List[str]:
		"""
		Lemmatizes tokens while excluding proper nouns, named entities, and financial terms.
		"""
		pos_tags = pos_tag(tokens)
		lemmatized_tokens = [
			self.lemmatizer.lemmatize(token, pos='v') if tag.startswith('V') and token.lower() not in self.financial_terms and token not in named_entities else token
			for token, tag in pos_tags
		]
		return lemmatized_tokens
	
	def extract_custom_patterns(self, text: str) -> dict:
		"""
		Extracts financial terms, time patterns, and currency values from text using regex patterns.
		"""
		extracted = {
			'financial_terms': [],
			'time_patterns': [],
			'currency_patterns': []
		}
		for category in ['financial_patterns', 'time_patterns', 'currency_patterns']:
			patterns = self.regex_patterns.get(category, [])
			for pattern in patterns:
				matches = re.findall(pattern, text)
				extracted[category.replace('_patterns', '')].extend(matches)
		return extracted
	
	def preprocess(self, text: str) -> str:
		"""
		Complete preprocessing pipeline for text.
		"""
		try:
			logger.info("Starting basic preprocessing of text.")
			# Extract dates and other custom patterns (optional step for date recognition and other financial/time data)
			custom_patterns = self.extract_custom_patterns(text)
			logger.debug(f"Extracted custom patterns: {custom_patterns}")
			# Lowercase text while preserving entities
			if self.config.get('clean', True):
				text = self.lowercase_text(text)
				logger.debug(f"After lowercasing: {text}")
			# Remove punctuation
			text = self.remove_punctuation(text)
			logger.debug(f"After punctuation removal: {text}")
			# Tokenize text
			tokens = self.tokenize_text(text)
			logger.debug(f"After tokenization: {tokens}")
			# Remove stopwords
			if self.config.get('remove_stopwords', True):
				tokens = self.remove_stopwords(tokens)
				logger.debug(f"After stopword removal: {tokens}")
			# Identify named entities
			named_entities = self.identify_named_entities(tokens)
			logger.debug(f"Identified named entities: {named_entities}")
			# Conditional lemmatization
			if self.config.get('lemmatize', True):
				tokens = self.conditional_lemmatization(tokens, named_entities)
				logger.debug(f"After lemmatization: {tokens}")
			# Return the preprocessed text as a string
			return ' '.join(tokens)
		except Exception as e:
			logger.error(f"Error during preprocessing: {str(e)}")
			return ""
		
if __name__ == "__main__":
	pass  # Placeholder for future standalone functionality or testing