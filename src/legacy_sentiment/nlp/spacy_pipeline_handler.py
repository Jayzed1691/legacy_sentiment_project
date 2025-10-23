#!/usr/bin/env python3

# spacy_pipeline_handler.py

import spacy
import re
import logging
from typing import List, Tuple, Dict, Any, Optional, Union, Set, NamedTuple
from legacy_sentiment.utils.custom_file_utils import load_json_file, load_language_data
from legacy_sentiment.processing.custom_entity_handler import CustomEntityHandler
from legacy_sentiment.processing.mwe_handler import MWEHandler
from legacy_sentiment.processing.regex_pattern_handler import RegexPatternHandler
from legacy_sentiment.utils.stopword_handler import StopwordHandler
from legacy_sentiment.nlp.semantic_role_handler import SemanticRoleHandler
from legacy_sentiment.utils.aspect_handler import AspectHandler
from legacy_sentiment.processing.text_cleaner import TextCleaner
from legacy_sentiment.processing.token_processor import TokenProcessor
from legacy_sentiment.data_models.data_types import EntityToken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class POSToken(NamedTuple):
	text: str
	lemma: str
	pos: str
	tag: str  # Detailed POS tag
	dep: str  # Dependency label
	start: int
	end: int
	
class NounChunk(NamedTuple):
	text: str
	root_text: str
	root_dep: str
	start: int
	end: int
	
class TextProcessor:
	"""Handle enhanced text processing."""
	def __init__(self, nlp: spacy.language.Language):
		self.nlp = nlp
		self.whitespace_pattern = re.compile(r'\s+')
		self.punctuation_pattern = re.compile(r'[^\w\s]')
		
	def get_protected_spans(self, doc: spacy.tokens.Doc) -> Set[Tuple[int, int]]:
		"""Get spans that should be protected from modification."""
		protected = set()
		
		# Protect named entities
		for ent in doc.ents:
			protected.add((ent.start_char, ent.end_char))
			
		# Protect noun chunks with important dependencies
		for chunk in doc.noun_chunks:
			if any(token.dep_ in {'nsubj', 'dobj', 'pobj'} for token in chunk):
				protected.add((chunk.start_char, chunk.end_char))
				
		return protected
	
	def clean_text(self, 
				text: str, 
				protected_spans: Set[Tuple[int, int]],
				normalize_whitespace: bool = True,
				remove_punctuation: bool = False) -> str:
		"""Clean text while preserving protected spans."""
		if not protected_spans:
			return self.whitespace_pattern.sub(' ', text).strip() if normalize_whitespace else text
		
		result = []
		last_end = 0
		
		for start, end in sorted(protected_spans):
			# Clean unprotected segment
			segment = text[last_end:start]
			if normalize_whitespace:
				segment = self.whitespace_pattern.sub(' ', segment)
			result.append(segment)
			
			# Add protected segment unchanged
			result.append(text[start:end])
			last_end = end
			
		# Clean final unprotected segment
		if last_end < len(text):
			segment = text[last_end:]
			if normalize_whitespace:
				segment = self.whitespace_pattern.sub(' ', segment)
			result.append(segment)
			
		return ''.join(result).strip()
	
class SpaCyPipelineHandler:
	def __init__(self, 
				language_data_files: Union[str, List[str]] = None,
				custom_entities_files: Union[str, List[str]] = None,
				mwe_files: Union[str, List[str]] = None,
				regex_patterns_files: Union[str, List[str]] = None,
				custom_stopwords_files: Union[str, List[str]] = None,
				aspect_config_files: Union[str, List[str]] = None,
				model: str = "en_core_web_sm"):
		"""
		Initialize pipeline with all handlers.
		
		Args:
			language_data_files: Language data file path(s)
			custom_entities_files: Custom entity definition file path(s)
			mwe_files: Multi-word expression file path(s)
			regex_patterns_files: Regex pattern file path(s)
			custom_stopwords_files: Custom stopword file path(s)
			model: SpaCy model to use
		"""
		# Initialize spaCy model
		try:
			self.nlp = spacy.load(model)
			logger.info(f"Loaded spaCy model: {model}")
		except OSError:
			self.nlp = spacy.load("en_core_web_sm")
			logger.warning(f"Model {model} not found, using en_core_web_sm instead")
			
		# Initialize semantic role handler
		self.semantic_role_handler = SemanticRoleHandler(self.nlp)
		
		# Initialize new text processing components
		self.text_processor = TextProcessor(self.nlp)
		
		# Load and verify language data
		self.language_data = load_language_data(language_data_files) if language_data_files else {}
		
		# Initialize all handlers
		self.custom_entity_handler = (CustomEntityHandler(custom_entities_files) 
									if custom_entities_files else None)
		self.mwe_handler = (MWEHandler(mwe_files) 
									if mwe_files else None)
		self.regex_handler = (RegexPatternHandler(regex_patterns_files) 
									if regex_patterns_files else None)
		self.stopword_handler = (StopwordHandler(custom_stopwords_files) 
									if custom_stopwords_files else None)
		self.token_processor = TokenProcessor(self.stopword_handler)
		
		# ADD aspect handler initialization
		if aspect_config_files:
			try:
				aspect_config = load_json_file(aspect_config_files[0])
				logger.info(f"Loaded aspect configuration with categories: {list(aspect_config.keys())}")
				self.aspect_handler = AspectHandler(self.nlp, aspect_config_files[0])
			except Exception as e:
				logger.warning(f"Failed to load aspect configuration: {e}")
				self.aspect_handler = None
		else:
				self.aspect_handler = None
			
		# Define valid POS tags for each lexical category
		self.CATEGORY_POS_MAPPING = {
			'modifiers': {
				'quantitative': {'ADV', 'ADP'},  # Added ADP for phrase components
				'specific': {'ADV', 'ADJ'}  
			},
			'uncertainty': {
				'modal': {'VERB', 'AUX'},  # Modal verbs
				'general': {'ADV', 'ADJ', 'VERB', 'NOUN'}  # Added NOUN for terms like "likelihood"
			},
			'negation': {
				'general': {'ADV', 'PART', 'INTJ', 'CCONJ', 'DET', 'PRON', 'NOUN'}  
				# Added CCONJ for "neither/nor", DET for determiners, PRON for pronouns
			},
			'causal': {
				'general': {'VERB', 'NOUN', 'ADJ'}  # Added NOUN and ADJ for words that can be both
			},
			'intensifiers': {'ADV'},  # Adverbs only
			'sarcasm': {
				'general': {'ADJ', 'ADV', 'INTJ', 'NOUN'},  # Added NOUN for terms like "genius"
				'phrase': {'INTJ'}  # Specific handling for sarcastic phrases
			}
		}
		
		# Initialize and verify lexical categories
		self._initialize_lexical_sets()
		
		# Existing patterns
		self.currency_pattern = re.compile(r'(\b(?:USD|Rp\.?|IDR|EUR|GBP|JPY)\s?)?(\d+(?:,\d{3})*(?:\.\d+)?)\s?(million|billion|trillion|rupiah)?', re.IGNORECASE)
		self.percentage_pattern = re.compile(r'(\d+(?:\.\d+)?)(?:(?:\s?)(?:-|–|—|to|\s)\s?)?(?:percent|%)', re.IGNORECASE)
		self.date_pattern = re.compile(r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b')
		self.time_comparison_pattern = re.compile(r'\b(year-on-year|month-on-month|quarter-on-quarter)\b', re.IGNORECASE)
		
		# POS tag sets for filtering
		self.RELEVANT_POS = {
			'ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB', 'NUM', 'AUX'  # Core POS tags
		}
		self.COMPLEX_NP_DEPS = {
			'compound', 'amod', 'nummod', 'nmod', 'npadvmod', 'prep', 'pobj'  # Dependencies for complex NPs
		}
		
		# Define source priorities as a class constant
		self.SOURCE_PRIORITIES = {
			'Regex': 0,        # Highest - most specific patterns
			'Custom': 1,       # Custom entities second
			'MWE': 2,         # Multi-word expressions third
			'spaCy': 3,       # Base spaCy entities last
			'default': 100    # Default priority for unknown sources
		}
		
	def _initialize_lexical_sets(self):
		"""Initialize lexical sets with POS validation rules."""
		def create_term_dict(category: str, terms: List[Dict]) -> Dict[str, Set[str]]:
			term_dict = {}
			for item in terms:
				term = item['term'].lower()
				# Get subcategory from the term's category list if available
				subcategory = next((cat for cat in item.get('category', []) 
									if cat in self.CATEGORY_POS_MAPPING.get(category, {})), 'general')
				
				# Handle both dictionary and set cases in POS mapping
				category_mapping = self.CATEGORY_POS_MAPPING.get(category, {})
				if isinstance(category_mapping, dict):
					term_dict[term] = category_mapping.get(subcategory, set())
				else:
					term_dict[term] = category_mapping  # If it's a set, use directly
					
			return term_dict
		
		# Convert plain sets to dictionaries mapping terms to their required POS tags
		self.quantitative_modifiers = create_term_dict('modifiers', 
			self.language_data.get('quantitative_modifiers', []))
		self.specific_modifiers = create_term_dict('modifiers',
			self.language_data.get('specific_modifiers', []))
		self.uncertainty_words = create_term_dict('uncertainty',
			self.language_data.get('uncertainty_words', []))
		self.negation_words = create_term_dict('negation',
			self.language_data.get('negation_words', []))
		self.causal_verbs = create_term_dict('causal',
			self.language_data.get('causal_verbs', []))
		self.intensifiers = create_term_dict('intensifiers',
			self.language_data.get('intensifiers', []))
		self.sarcasm_features = create_term_dict('sarcasm',
			self.language_data.get('sarcasm_features', []))
		
	def _remove_overlaps(self, 
			entities: List[EntityToken], 
			strategy: str = 'priority') -> List[EntityToken]:
		"""
		Remove overlapping entities using configurable strategies.
			
		Args:
			entities: List of EntityToken objects
			strategy: Conflict resolution strategy ('priority', 'length', or 'position')
			
		Returns:
			List of non-overlapping EntityTokens
		"""
		if not entities:
			return []
		
		def get_sort_key(entity: EntityToken) -> Tuple:
			if strategy == 'priority':
				return (
					entity.start,
					-len(entity.text),  # Longer matches first
					self.SOURCE_PRIORITIES.get(entity.source, 
						self.SOURCE_PRIORITIES['default'])
				)
			elif strategy == 'length':
				return (
					entity.start,
					-len(entity.text)  # Longer matches first
				)
			else:  # position strategy
				return (entity.start, entity.end)
		
		# Sort entities using strategy
		sorted_entities = sorted(entities, key=get_sort_key)
		
		# Track covered spans for efficient overlap checking
		covered_spans: List[Tuple[int, int]] = []
		non_overlapping = []
		
		def spans_overlap(span1: Tuple[int, int], 
						span2: Tuple[int, int]) -> bool:
			return (span1[0] < span2[1] and span2[0] < span1[1])
		
		for entity in sorted_entities:
			entity_span = (entity.start, entity.end)
			
			# Check if this entity overlaps with any covered span
			if not any(spans_overlap(entity_span, span) 
						for span in covered_spans):
				non_overlapping.append(entity)
				covered_spans.append(entity_span)
			
		return non_overlapping
	
	def _convert_to_entity_tokens(self, 
			entities: List[Tuple], 
			source: str,
			doc: spacy.tokens.Doc) -> List[EntityToken]:
		"""Convert tuple entities to EntityTokens with POS info."""
		return [
			EntityToken(
				text=ent[0],
				label=ent[1],
				source=source,
				start=ent[3],
				end=ent[4],
				pos_tag=self._get_span_pos(doc, ent[3], ent[4])
			)
			for ent in entities
		]
	
	def _get_span_pos(self, 
			doc: spacy.tokens.Doc,
			start: int,
			end: int) -> Optional[str]:
		"""Get POS tag for a span."""
		span = doc.char_span(start, end)
		if span:
			return span.root.pos_
		return None
	
	def analyze_text(self, text: str) -> Dict[str, Any]:
		"""
		Perform comprehensive text analysis with all handlers and processors.
		
		Args:
			text: Input text to analyze
			
		Returns:
			Dictionary containing all analysis results including:
			- entities: List of identified entities from all sources
			- processed_tokens: List of processed tokens with provenance
			- protected_spans: List of spans to protect from modification
			- pos_tags: Part of speech analysis
			- noun_chunks: Extracted noun phrases
			- lexical_features: Identified lexical patterns
			- semantic_roles: Semantic role labeling results
			- aspects: Extracted aspects and their targets
		"""
		# Create spaCy doc once
		doc = self.nlp(text)
		
		# Get base spaCy analysis
		spacy_analysis = self.analyze_text_from_doc(doc)
		
		# Get semantic roles
		semantic_roles = self.semantic_role_handler.extract_roles_from_doc(doc)
		predicate_args = self.semantic_role_handler.get_predicate_argument_structure_from_doc(doc)
		
		# ADD aspect analysis
		if self.aspect_handler:
			aspects = self.aspect_handler.extract_aspects_from_doc(
				doc=doc,
				noun_chunks=spacy_analysis['noun_chunks'],
				semantic_roles=semantic_roles
			)
		else:
			aspects = []
			
		# Collect entities from all sources with priority handling
		all_entities = []
		
		# Regex patterns (highest priority)
		if self.regex_handler:
			regex_ents = self.regex_handler.process_text(text)
			regex_tokens = self._convert_to_entity_tokens(
				regex_ents, 
				'Regex',
				doc
			)
			all_entities.extend(regex_tokens)
			
		# Custom entities (second priority)
		if self.custom_entity_handler:
			custom_ents = self.custom_entity_handler.extract_named_entities(text)
			custom_tokens = self._convert_to_entity_tokens(
				custom_ents,
				'Custom',
				doc
			)
			all_entities.extend(custom_tokens)
			
		# Multi-word expressions (third priority)
		if self.mwe_handler:
			mwe_ents = self.mwe_handler.extract_multi_word_expressions(text)
			mwe_tokens = self._convert_to_entity_tokens(
				mwe_ents,
				'MWE',
				doc
			)
			all_entities.extend(mwe_tokens)
			
		# spaCy entities (lowest priority)
		spacy_ents = self._convert_to_entity_tokens(
			spacy_analysis['entities'],
			'spaCy',
			doc
		)
		all_entities.extend(spacy_ents)
		
		# Resolve overlaps using priority strategy
		final_entities = self._remove_overlaps(all_entities, strategy='priority')
		
		# Get semantic roles
		semantic_roles = self.semantic_role_handler.extract_roles_from_doc(doc)
		predicate_args = self.semantic_role_handler.get_predicate_argument_structure_from_doc(doc)
		
		# Process tokens with entity protection
		if self.token_processor:
			processed_tokens = self.token_processor.process_text_from_doc(
				doc, final_entities)
			
			# Generate protected spans from entities and important semantic roles
			protected_spans = set()
			
			# Add entity spans
			for entity in final_entities:
				protected_spans.add((entity.start, entity.end))
				
			# Add core semantic role spans
			for role in semantic_roles:
				if role.role in {'AGENT', 'PATIENT', 'RECIPIENT'}:
					protected_spans.add((role.start, role.end))
					
			# Sort spans for consistent output
			protected_spans = list(sorted(protected_spans))
		else:
			processed_tokens = []
			protected_spans = []
			
		# Compile results
		results = {
			# Core analysis
			'entities': final_entities,
			'processed_tokens': processed_tokens,
			'protected_spans': protected_spans,
			
			# SpaCy analysis
			'pos_tags': spacy_analysis['pos_tags'],
			'noun_chunks': spacy_analysis['noun_chunks'],
			'lexical_features': spacy_analysis['lexical_features'],
			
			# Semantic analysis
			'semantic_roles': semantic_roles,
			'predicate_arguments': predicate_args,
			'aspects': aspects,
			
			# Source-specific entities (for reference)
			'spacy_entities': spacy_analysis['entities'],
			'custom_entities': custom_ents if 'custom_ents' in locals() else [],
			'mwe': mwe_ents if 'mwe_ents' in locals() else [],
			'regex_matches': regex_ents if 'regex_ents' in locals() else [],
		}
		
		# Add refined entities if available
		if 'refined_entities' in spacy_analysis:
			results['refined_entities'] = spacy_analysis['refined_entities']
			
		# Add stopword analysis if available
		if self.stopword_handler:
			results['stopwords'] = self.stopword_handler.process_text_from_doc(doc)
			
		return results
	
	def get_active_handlers(self) -> Dict[str, bool]:
		"""Return status of all handlers."""
		return {
			'custom_entities': bool(self.custom_entity_handler),
			'mwe': bool(self.mwe_handler),
			'regex_patterns': bool(self.regex_handler),
			'stopwords': bool(self.stopword_handler),
			'language_data': bool(self.language_data),
			'semantic_roles': bool(self.semantic_role_handler),
			'aspect_handler': bool(self.aspect_handler)
		}
	
	def analyze_text_from_doc(self, doc: spacy.tokens.Doc) -> Dict[str, Any]:
		"""New method - analyze pre-processed spaCy Doc object"""
		# Get entities first
		entities = self.extract_spacy_entities(doc)
		
		return {
			'pos_tags': self.get_pos_tags(doc),
			'noun_chunks': self.extract_noun_chunks(doc),
			'entities': self.extract_spacy_entities(doc),
			'lexical_features': self._analyze_lexical_features(doc),
			'refined_entities': self.refine_entities(doc, entities)
		}
	
	def _analyze_lexical_features(self, doc: spacy.tokens.Doc) -> Dict[str, List[Dict[str, Any]]]:
		"""
		Analyze text for lexical features with POS verification.
		"""
		# Pre-tokenize the text to get word boundaries, POS tags, and entities
		text = doc.text 
		text_lower = text.lower()
		
		# Create entity span lookup for quick verification
		entity_spans = {(ent.start_char, ent.end_char): ent.label_ for ent in doc.ents}
		
		matches = {
			'modifiers': [],
			'uncertainty': [],
			'negation': [],
			'causal': [],
			'intensifiers': [],
			'sarcasm': []
		}
		
		def is_complete_match(term: str, start: int, text: str) -> bool:
			end = start + len(term)
			before_ok = start == 0 or not text[start - 1].isalnum()
			after_ok = end == len(text) or not text[end].isalnum()
			return before_ok and after_ok
		
		def get_token_at_position(char_pos: int) -> Optional[spacy.tokens.Token]:
			for token in doc:
				if token.idx <= char_pos < token.idx + len(token.text):
					return token
			return None
		
		def is_part_of_entity(start: int, end: int) -> Optional[str]:
			"""Check if span is part of a named entity."""
			for (ent_start, ent_end), label in entity_spans.items():
				if (start >= ent_start and end <= ent_end):
					return label
			return None
		
		def is_valid_match(token: spacy.tokens.Token, term: str, 
							category: str, start: int) -> bool:
			"""Validate match based on POS tags and entity information."""
			end = start + len(term)
			# Check if part of a named entity
			entity_type = is_part_of_entity(start, end)
			if entity_type:
				return False
			
			# Get the term's required POS tags based on category
			term_lower = term.lower()
			required_pos = set()
			
			if category == 'modifiers':
				if term_lower in self.quantitative_modifiers:
					required_pos = self.quantitative_modifiers[term_lower]
				elif term_lower in self.specific_modifiers:
					required_pos = self.specific_modifiers[term_lower]
			elif category == 'uncertainty':
				if term_lower in self.uncertainty_words:
					required_pos = self.uncertainty_words[term_lower]
			elif category == 'negation':
				if term_lower in self.negation_words:
					required_pos = self.negation_words[term_lower]
			elif category == 'causal':
				if term_lower in self.causal_verbs:
					required_pos = self.causal_verbs[term_lower]
			elif category == 'intensifiers':
				if term_lower in self.intensifiers:
					required_pos = self.intensifiers[term_lower]
			elif category == 'sarcasm':
				if term_lower in self.sarcasm_features:
					required_pos = self.sarcasm_features[term_lower]
					
			# If we have POS requirements, check them
			if required_pos:
				return token.pos_ in required_pos
			
			return False  # If no valid POS mapping found, reject the match
		
		def find_all_matches(term: str) -> List[Tuple[int, int, spacy.tokens.Token]]:
			matches = []
			start = 0
			while True:
				start = text_lower.find(term, start)
				if start == -1:
					break
				if is_complete_match(term, start, text_lower):
					token = get_token_at_position(start)
					if token:
						matches.append((start, start + len(term), token))
				start += 1
			return matches
		
		def add_match(category: str, term: str, match_type: str, 
					positions: List[Tuple[int, int, spacy.tokens.Token]]):
			for start, end, token in positions:
				if is_valid_match(token, term, category, start):
					ent_type = is_part_of_entity(start, end)
					matches[category].append({
						'term': text[start:end],  # Original case
						'type': match_type,
						'position': (start, end),
						'pos_tag': token.pos_,  # Changed from 'pos' to 'pos_tag'
						'entity_type': ent_type
					})
					
		# Process each lexical category
		term_categories = [
			(self.quantitative_modifiers, 'modifiers', 'quantitative'),
			(self.specific_modifiers, 'modifiers', 'specific'),
			(self.uncertainty_words, 'uncertainty', 'uncertainty'),
			(self.negation_words, 'negation', 'negation'),
			(self.causal_verbs, 'causal', 'causal'),
			(self.intensifiers, 'intensifiers', 'intensifier'),
			(self.sarcasm_features, 'sarcasm', 'sarcasm')
		]
		
		for term_set, category, match_type in term_categories:
			for term in term_set:
				positions = find_all_matches(term)
				if positions:
					add_match(category, term, match_type, positions)
					
		# Sort matches within each category by position
		for category in matches:
			matches[category].sort(key=lambda x: x['position'][0])
			
		return matches
	
	def get_pos_tags(self, doc: spacy.tokens.Doc) -> List[POSToken]:
		"""Get enhanced POS tagging with detailed linguistic information."""
		return [
			POSToken(
				text=token.text,
				lemma=token.lemma_,
				pos=token.pos_,
				tag=token.tag_,
				dep=token.dep_,
				start=token.idx,
				end=token.idx + len(token.text)
			)
			for token in doc
			if token.pos_ in self.RELEVANT_POS
		]
	
	def extract_noun_chunks(self, doc: spacy.tokens.Doc) -> List[NounChunk]:
		"""Extract and analyze noun chunks with their internal structure."""
		return [
			NounChunk(
				text=chunk.text,
				root_text=chunk.root.text,
				root_dep=chunk.root.dep_,
				start=chunk.start_char,
				end=chunk.end_char
			)
			for chunk in doc.noun_chunks
			if self._is_valid_chunk(chunk)
		]
	
	def _is_valid_chunk(self, chunk: spacy.tokens.Span) -> bool:
		"""Validate noun chunks based on internal structure and dependencies."""
		has_modifiers = any(token.dep_ in self.COMPLEX_NP_DEPS for token in chunk)
		if len(chunk) == 1 and chunk[0].pos_ == 'NOUN' and not has_modifiers:
			return False
		return True
	
	def _contains_and(self, text: str) -> bool:
		return bool(re.search(r'\band\b', text, re.IGNORECASE))
	
	def refine_entities(self, doc: spacy.tokens.Doc, entities: List[Tuple[str, str, str, int, int]]) -> List[Tuple[str, str, str, int, int]]:
		"""Refine entities using pre-processed Doc."""
		refined_entities = []
		for entity in entities:
			entity_text, entity_type, source, start, end = entity
			
			# Discard entities containing "and"
			if self._contains_and(entity_text):
				continue
			
			# Check for quantitative modifiers at the beginning of the entity
			modifier_match = None
			for modifier in self.quantitative_modifiers:
				if entity_text.lower().startswith(modifier + " "):
					modifier_match = modifier
					break
				
			if modifier_match:
				modifier_end = start + len(modifier_match)
				refined_entities.append((modifier_match, 'QUANTITATIVE_MODIFIER', 'Refined', start, modifier_end))
				start = modifier_end + 1
				entity_text = entity_text[len(modifier_match)+1:].strip()
				
			# Proceed with existing refinement logic
			if self.date_pattern.search(entity_text):
				refined_entities.append((entity_text, 'DATE', 'Refined', start, end))
			elif self.percentage_pattern.search(entity_text):
				refined_entities.append((entity_text, 'PERCENT', 'Refined', start, end))
			elif self.currency_pattern.search(entity_text):
				currency_match = self.currency_pattern.search(entity_text)
				currency, amount, unit = currency_match.groups()
				if currency or unit:
					refined_entities.append((entity_text, 'MONEY', 'Refined', start, end))
				else:
					refined_entities.append((entity_text, 'NUMBER', 'Refined', start, end))
			elif self.time_comparison_pattern.search(entity_text):
				refined_entities.append((entity_text, 'TIME_COMPARISON', 'Refined', start, end))
			else:
				refined_entities.append((entity_text, entity_type, source, start, end))
				
		return refined_entities
	
	def extract_spacy_entities(self, doc: spacy.tokens.Doc) -> List[Tuple[str, str, str, int, int]]:
		"""Extract entities using pre-processed Doc."""
		return [(ent.text, ent.label_, 'spaCy', ent.start_char, ent.end_char) 
				for ent in doc.ents if not self._contains_and(ent.text)]
	
	def get_active_handlers(self) -> Dict[str, bool]:
		"""Return status of all handlers."""
		return {
			'custom_entities': bool(self.custom_entity_handler),
			'mwe': bool(self.mwe_handler),
			'regex_patterns': bool(self.regex_handler),
			'stopwords': bool(self.stopword_handler),
			'language_data': bool(self.language_data)
		}