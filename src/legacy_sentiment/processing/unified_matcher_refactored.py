#!/usr/bin/env python3

# unified_matcher_refactored.v.1120.1.py

import re
from typing import List, Dict, Any, Tuple, Optional, Union, Set
import logging
from ahocorasick import Automaton
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span
from legacy_sentiment.data_models.data_types import (
	Token,
	SPACY_MODEL_SM,
	SPACY_MODEL_MD,
	SPACY_MODEL_LG,
	SPACY_MODEL_TRF,
	SPACY_MODEL_DEPENDENCIES,
)

# Configure the logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Default SpaCy model (loaded on demand)
NLP = None

def set_spacy_model(model_name: str):
	"""Allows switching SpaCy model dynamically."""
	global NLP
	NLP = spacy.load(model_name)
	
def expand_span(doc: Doc, span: Span, expand_trailing: bool = True) -> Span:
	"""Expands a span to include related modifiers based on dependency labels."""
	"""
	start, end = span.start, span.end
	model_name = NLP.meta["name"]
	deps = SPACY_MODEL_DEPENDENCIES.get(model_name, SPACY_MODEL_DEPENDENCIES[SPACY_MODEL_SM])

	# Expand start position
	while start > 0 and doc[start - 1].dep_ in deps["preceding"] and doc[start - 1].dep_ not in {"det", "prep"}:
		start -= 1

	# Expand end position if allowed
	if expand_trailing:
		while end < len(doc) and doc[end].dep_ in deps["trailing"]:
			end += 1

	return doc[start:end]
	"""
	return span

def refine_entities(
	doc: Doc,
	entities: List[Tuple[str, str, str, int, int]],
	expand_spans: bool = False,
	expand_trailing: bool = False,
) -> List[Token]:
	"""Refine entities without unnecessary splitting or expansion."""
	refined_entities = []
	for entity_text, entity_label, source, start, end in entities:
		# Create a span directly
		refined_entities.append(
			create_token(
				original_text=entity_text,
				start=start,
				end=end,
				source=source,
				category="Entity",
				label=entity_label,
				doc=doc,
				expand_trailing=expand_trailing,
			)
		)
	return refined_entities

def get_excluded_positions(matches: List[Token]) -> set:
	"""Create a set of excluded positions from matches."""
	return set(pos for match in matches for pos in range(match.start, match.end))

def is_position_excluded(start: int, end: int, excluded_positions: set) -> bool:
	"""Check if a token's character range overlaps with excluded positions."""
	return any(pos in excluded_positions for pos in range(start, end))

def create_token(
	original_text: str,
	start: int,
	end: int,
	source: str,
	category: Union[str, List[str]],
	label: str,
	doc: Optional[Doc] = None,
	expanded_text: Optional[str] = None,
	expand_trailing: bool = False,
	pos_tag: Optional[str] = None,
	lemma: Optional[str] = None,
) -> Token:
	"""
	Creates a Token with consistent rules for expansion and lemmatization.
	Handles category as both a string or a list of strings.

	Args:
		original_text: The original matched text
		start: Start character position
		end: End character position
		source: Source of the match (Entity, MWE, Regex, SpaCy)
		category: Category of the match
		label: Label/type of the match
		doc: Optional spaCy Doc for span-based operations
		expanded_text: Pre-computed expanded text (optional)
		expand_trailing: Whether to expand trailing tokens
		pos_tag: The POS tag to assign to the token (optional)
		lemma: The lemma of the token (optional)
	"""
	
	# Convert the category to a string if it's a list for consistent handling
	category_str = category[-1] if isinstance(category, list) else category
	
	# Don't process lemmas/POS for any of these cases
	skip_linguistic_processing = (
		source in {"Entity", "MWE", "Regex"}
		or label in {"Entity", "Noun Chunk"}
		or category_str in {"Entity", "Noun Chunk"}
		or source != "SpaCy"
	)
	
	# Handle span operations if doc is provided
	span = None
	if doc:
		span = doc.char_span(start, end, alignment_mode="expand")
		if expand_trailing and span:
			span = expand_span(doc, span)
			
	# Set expanded text
	final_expanded_text = (
		expanded_text or (span.text if span else None) or original_text
	)
	
	# Use provided POS tag and lemma if available, otherwise compute them
	if not skip_linguistic_processing:
		final_pos_tag = pos_tag if pos_tag is not None else (span.root.pos_ if span and span.root else None)
		final_lemma = lemma if lemma is not None else (span.root.lemma_ if span and span.root else None)
	else:
		final_pos_tag = pos_tag if pos_tag is not None else None
		final_lemma = lemma if lemma is not None else None
		
	#Pass the spaCy token here
	final_token = span.root if span else None
	
	return Token(
		original_text=original_text,
		expanded_text=final_expanded_text,
		source=source,
		category=category_str,
		label=label,
		start=start,
		end=end,
		pos_tag=pos_tag,
		lemma=lemma,
		token=final_token,
	)
	
def get_stopword_status(
	token_text: str,
	token,
	custom_stopwords: Optional[set],
	include_stopwords: bool,
) -> bool:
	"""Determine if a token is a stopword based on custom or SpaCy-defined stopwords."""
	is_custom_stopword = custom_stopwords and token_text in custom_stopwords
	is_spacy_stopword = token.is_stop
	
	if include_stopwords:
		return is_custom_stopword or is_spacy_stopword
	else:
		return is_custom_stopword
	
def sort_and_deduplicate_matches(matches: List[Token]) -> List[Token]:
	"""Sorts and removes overlapping matches."""
	matches.sort(key=lambda x: (x.start, -len(x.original_text)))
	deduplicated = []
	last_end = -1
	for match in matches:
		if match.start >= last_end:
			deduplicated.append(match)
			last_end = match.end
	return deduplicated

def is_word_boundary(text: str, start: int, end: int) -> bool:
	"""Check if a match is at word boundaries."""
	before_ok = start == 0 or not text[start - 1].isalnum()
	after_ok = end == len(text) or not text[end:end + 1].isalnum()  # Corrected
	return before_ok and after_ok

def match_entities(
	text: str,
	doc: Doc,
	entities: Dict[str, Automaton],
	expand_trailing: bool,
) -> List[Token]:
	"""Match entities using automaton, including variations."""
	matches = []
	for category, entity_automaton in entities.items():
		for end_idx, payload in entity_automaton.iter(text.lower()):
			matched_term = payload.get("full_name") or payload.get("term")
			variations = payload.get("variations", [])
			categories = payload.get("category", [])
			
			# Extract linguistic information from payload
			pos_tag = payload.get("pos")
			lemma = payload.get("lemma")
			
			for term in [matched_term] + variations:
				if term:
					start_idx = text.lower().find(term.lower(), end_idx - len(term) + 1)
					if start_idx != -1 and is_word_boundary(
						text, start_idx, start_idx + len(term)
					):
						matches.append(
							create_token(
								original_text=term,
								start=start_idx,
								end=start_idx + len(term),
								source="Entity",
								category=category,
								label=category,
								doc=doc,
								expand_trailing=expand_trailing,
								pos_tag=pos_tag,
								lemma=lemma,
							)
						)
	return matches

def match_mwes(
	text: str,
	doc: Doc,
	mwes: Dict[str, Automaton],
	expand_trailing: bool,
) -> List[Token]:
	"""Match MWEs using automaton."""
	matches = []
	text_lower = text.lower()  # Lowercase the text once for efficiency
	
	for category, mwe_automaton in mwes.items():
		for end_idx, payload in mwe_automaton.iter(text_lower):
			# Access the 'term' and 'variations' from the payload
			term = payload.get("term")
			variations = payload.get("variations", [])
			pos = payload.get("pos")
			lemma = payload.get("lemma")
			
			# Consider the term and all its variations
			for current_term in [term] + variations:
				start_idx = text_lower.find(current_term.lower(), 0, end_idx + 1)
				if start_idx != -1:
					end_idx = start_idx + len(current_term)
					if is_word_boundary(text, start_idx, end_idx):
						matches.append(create_token(
							original_text=current_term,
							start=start_idx,
							end=end_idx,
							source="MWE",
							category=category,
							label=category,
							doc=doc,
							expand_trailing=expand_trailing,
							pos_tag = pos,
							lemma = lemma
						))
	return matches

def match_regex_patterns(
	text: str,
	doc: Doc,
	regex_patterns: Dict[str, List[Dict[str, Any]]],
	expand_trailing: bool,
) -> List[Token]:
	"""Matches regex patterns in the text and returns a list of Token objects."""
	matches = []
	for category, patterns in regex_patterns.items():
		for pattern_info in patterns:
			pattern = pattern_info.get("pattern")
			label = pattern_info.get("label", "Regex")
			if not pattern:
				continue
			for match in re.finditer(pattern, text):
				start_idx, end_idx = match.span()
				term = match.group()
				# Create a span from the matched text
				matches.append(
					create_token(
						original_text=term,
						start=start_idx,
						end=end_idx,
						source="Regex",
						category=category,
						label=label,
						doc=doc,
						expand_trailing=expand_trailing,
						lemma=term  # Set lemma to the matched term for regex
					)
				)
	return matches

def extract_spacy_entities(doc: Doc) -> List[Tuple[str, str, str, int, int]]:
	"""Extract entities using SpaCy."""
	return [
		(ent.text, ent.label_, "SpaCy", ent.start_char, ent.end_char) for ent in doc.ents
	]
	
def process_spacy_matches(
	doc: Doc,
	excluded_positions: set,
	min_noun_chunk_length: int = 1,
	custom_stopwords: Optional[set] = None,
	include_stopwords: bool = False,
) -> List[Token]:
	"""Processes SpaCy entities, noun chunks, and remaining tokens, excluding overlaps with previously matched positions."""
	spacy_matches = []
	
	# Step 1: Extract entities
	raw_entities = extract_spacy_entities(doc)
	
	# Step 2: Filter out entities that overlap with excluded positions
	filtered_entities = [
		entity
		for entity in raw_entities
		if not is_position_excluded(entity[3], entity[4], excluded_positions)
	]
	
	# Step 3: Refine entities without span expansion
	refined_entities = refine_entities(
		doc,
		filtered_entities,
		expand_spans=False,
		expand_trailing=False,
	)
	
	# Add entities to spacy_matches list
	for token in refined_entities:
		spacy_matches.append(
			create_token(
				original_text=token.original_text,
				start=token.start,
				end=token.end,
				source=token.source,
				category="Entity",
				label=token.label,
				doc=doc,
				expanded_text=token.expanded_text,
			)
		)
		
	# Update excluded_positions set to include entity positions
	entity_positions = set()
	for token in spacy_matches:
		entity_positions.update(range(token.start, token.end))
	excluded_positions.update(entity_positions)
	
	# Step 4: Process noun chunks
	for chunk in doc.noun_chunks:
		if len(chunk) >= min_noun_chunk_length and not is_position_excluded(
			chunk.start_char, chunk.end_char, excluded_positions
		):
			# Get the root of the noun chunk for lemma and POS
			root = chunk.root
			spacy_matches.append(
				create_token(
					original_text=chunk.text,
					start=chunk.start_char,
					end=chunk.end_char,
					source="SpaCy",
					category="Noun Chunk",
					label="Noun Chunk",
					doc=doc,
					pos_tag=root.pos_,  # Get POS from the root
					lemma=root.lemma_   # Get lemma from the root
				)
			)
			# Update excluded positions for noun chunks
			excluded_positions.update(range(chunk.start_char, chunk.end_char))
		
	# Step 5: Process remaining tokens
	for token in doc:
		token_start = token.idx
		token_end = token.idx + len(token.text)
		
		# Skip if token position overlaps with any previous matches
		if is_position_excluded(token_start, token_end, excluded_positions):
			continue
		
		token_text = token.text.lower()  # Lowercase the token text
		
		# Check stopword status using the helper function
		is_stopword = get_stopword_status(token_text, token, custom_stopwords, include_stopwords)
		
		if is_stopword:
			# Determine source of stopword (SpaCy or Custom)
			if custom_stopwords and token_text in custom_stopwords:
				source = "Custom"
			else:
				source = "SpaCy"
			label = "Stopword"
		else:
			source = "SpaCy"
			label = token.pos_
			
		spacy_matches.append(
			create_token(
				original_text=token.text,
				start=token_start,
				end=token_end,
				source=source,
				category=label,
				label=label,
				doc=doc,
				expanded_text=token.text,  # No expansion for individual tokens
				pos_tag=token.pos_,  # Add POS tag
				lemma=token.lemma_  # Add lemma
			)
		)
		# Update excluded positions for this token
		excluded_positions.update(range(token_start, token_end))
		
	return spacy_matches

def create_match_info(token, pattern, source, start_idx, end_idx):
	"""Helper function to create match info dictionary."""
	category = pattern.get("category", [])
	if isinstance(category, list):
		category = ", ".join(category)  # Convert list to string
		
	return {
		"original_text": token.text,
		"lemma": token.lemma_,
		"pos_tag": token.pos_,
		"start": start_idx,
		"end": end_idx,
		"category": category,  # Use the string representation of the category
		"label": category,
		"source": source,
	}
	
def match_language_patterns(
	doc: Doc,
	language_automaton: Automaton,
	entities: Dict[str, Automaton],
	mwes: Dict[str, Automaton],
) -> List[Token]:
	"""
	Matches language patterns using Aho-Corasick automaton, lemmas, and dependency parsing.
	This version accommodates the new automaton structure and the updated custom_language_data.json.
	"""
	matches = []
	text_lower = doc.text.lower()  # Use lowercase for case-insensitive matching
	
	for end_idx, (normalized_term, pattern_data) in language_automaton.iter(
		text_lower
	):
		# Extract information from the pattern_data
		term = pattern_data["term"]
		lemma = pattern_data["lemma"]
		category = pattern_data["category"]  # This is now a list
		pos_list = pattern_data.get("pos", [])  # Handle cases where 'pos' might be missing
		weight = pattern_data.get("weight")
	
		# Find the start index of the matched term
		start_idx = text_lower.rfind(normalized_term, 0, end_idx + 1)
		if start_idx == -1:
			continue  # Skip if term not found
	
		end_idx = start_idx + len(normalized_term)  # Adjust end_idx to the actual end of the term
	
		# Check if the match is at word boundaries
		if not is_word_boundary(text_lower, start_idx, end_idx):
			continue
	
		# Iterate over tokens in the doc
		for token in doc:
			token_start = token.idx
			token_end = token_start + len(token.text)
			
			# Check if this token overlaps with the matched pattern
			if token_start >= start_idx and token_end <= end_idx:
				term_lower = term.lower()
				
				# Check if the token's text or lemma matches the term or any variation
				if (
					token.text.lower() == term_lower
					or token.lemma_.lower() == term_lower
				):
					# Create the match_info with the extracted details
					match_info = {
						"original_text": token.text,
						"lemma": lemma,
						"pos_tag": token.pos_,  # Use the token's POS tag
						"start": start_idx,
						"end": end_idx,
						"category": category,  # Use the category list
						"label": category[-1]
						if category
						else "LanguagePattern",  # Use the last part of the category as label or default
						"source": "LangPat",
					}
					matches.append(create_token(**match_info, doc=doc))
					break
			
	return matches

def get_categories(
	token: spacy.tokens.Token,
	entities: Dict[str, Automaton],
	mwes: Dict[str, Automaton],
) -> List[str]:
	"""
	Determines the categories a token belongs to based on entity, MWE, or language pattern matches.

	Args:
		token: The spaCy Token to check.
		entities: The entities automaton.
		mwes: The MWEs automaton.

	Returns:
		A list of categories the token belongs to.
	"""
	categories = []
	
	# Check if the token is part of an entity
	for category, entity_automaton in entities.items():
		# Iterate over matches in the automaton for the token's text
		for _, payload in entity_automaton.iter(token.text.lower()):
			categories.extend(payload.get("category", []))
			
	# Check if the token is part of an MWE
	for category, mwe_automaton in mwes.items():
		# Iterate over matches in the automaton for the token's text
		for _, payload in mwe_automaton.iter(token.text.lower()):
			categories.extend(payload.get("category", []))
			
	# Optionally, add logic to check if the token is part of a language pattern
	# This will depend on how you decide to store and access language pattern matches
			
	return list(set(categories))  # Return unique categories

def unified_match(
	text: str,
	entities: Dict[str, Automaton],
	mwes: Dict[str, Automaton],
	regex_patterns: Dict[str, List[Dict[str, Any]]],
	custom_stopwords: Optional[set] = None,
	include_stopwords: bool = False,
	min_noun_chunk_length: int = 1,
	language_automaton: Optional[Automaton] = None,
	expand_trailing: bool = False,
) -> List[Token]:
	"""Unified matching pipeline for entities, MWEs, regex, SpaCy tokens and language patterns."""
	global NLP
	if NLP is None:
		try:
			NLP = spacy.load(SPACY_MODEL_SM)
		except OSError:
			logger.warning(f"spaCy model {SPACY_MODEL_SM} not found. Some features may not work.")
			# Return minimal matches without spaCy
			return []
	doc = NLP(text)
	matches = []
	
	print("Language Automaton in unified_match:", language_automaton)
	
	# Step 1: Match entities
	matches.extend(match_entities(text, doc, entities, expand_trailing))
	
	# Step 2: Match MWEs
	mwe_matches = match_mwes(text, doc, mwes, expand_trailing)
	matches.extend(mwe_matches)
	
	# Step 3: Match regex patterns
	matches.extend(match_regex_patterns(text, doc, regex_patterns, expand_trailing))
	
	# Step 4: Match language patterns
	if language_automaton:
		print("Language Automaton is not None. Proceeding to match language patterns.")
		language_matches = match_language_patterns(
			doc, language_automaton, entities, mwes
		)
		matches.extend(language_matches)
	else:
		print("Language Automaton is None. Skipping language pattern matching.")
		
	# Step 5: Calculate excluded positions based on all matches so far
	excluded_positions = get_excluded_positions(matches)
	
	# Step 6: Process SpaCy entities, noun chunks, and remaining tokens
	spacy_matches = process_spacy_matches(
		doc=doc,
		excluded_positions=excluded_positions,
		min_noun_chunk_length=min_noun_chunk_length,
		custom_stopwords=custom_stopwords,
		include_stopwords=include_stopwords,
	)
	matches.extend(spacy_matches)
	
	# Step 7: Sort and deduplicate matches, prioritizing longer matches
	matches = sort_and_deduplicate_matches(matches)
	
	return matches