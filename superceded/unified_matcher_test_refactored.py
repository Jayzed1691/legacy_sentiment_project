#!/usr/bin/env python3

# unified_matcher_test_refactored.py

import logging
import streamlit as st
import pandas as pd
import copy
import spacy
from typing import Dict, List, Set, Any, Optional, Union
from transcript_upload import upload_transcript_files
from config_file_upload import upload_custom_files
from config_custom_load import load_configs
from transcript_handler import TranscriptHandler
from unified_matcher_refactored import unified_match, set_spacy_model, NLP
from transcript_display import display_transcripts
from text_cleaner import clean_matched_text
from enhanced_semantic_role_handler import (
	EnhancedSemanticRoleHandler,
	EnhancedSemanticRole,
	SemanticComplex,
)
from semantic_role_config import CertaintyType, CertaintyDegree, CertaintyAssessment
from data_types import (
	SPACY_MODEL_SM,
	SPACY_MODEL_MD,
	SPACY_MODEL_LG,
	SPACY_MODEL_TRF,
	SPACY_MODEL_INFO,
	Token,
	SemanticRole,
)
from enhanced_semantic_display_handler import SemanticDisplayHandler

# Configure the logger
logging.basicConfig(
	level=logging.DEBUG,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
	handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Set the Streamlit app layout to wide
st.set_page_config(layout="wide")

class TranscriptMatchingTestWrapper:
	def __init__(self):
		"""Initialize the test wrapper and load configurations from uploaded files."""
		self.config_files = upload_custom_files()
		self.configs = load_configs(self.config_files)
		self.transcript_handler = TranscriptHandler()
		set_spacy_model(SPACY_MODEL_SM)
		# Extract relevant patterns from loaded language data
		self.language_automaton = self.configs.get("language_data", None)  # Store as class member
		self.semantic_handler = EnhancedSemanticRoleHandler(
			NLP, semantic_role_data=self.configs.get("semantic_roles")
		)
		self.display_handler = SemanticDisplayHandler(NLP)
		self.processed_transcripts = None
		self.matches_processed = False
		self.filtered_transcripts = None
		self.unfiltered_transcripts = None
		
		# Initialize session state variables if they don't exist
		if "unfiltered_transcripts" not in st.session_state:
			st.session_state["unfiltered_transcripts"] = None
		if "matches_processed" not in st.session_state:
			st.session_state["matches_processed"] = False
			
	def _format_analysis_for_display(
		self,
		matches: List[Token],
		semantic_roles: Optional[List[Union[EnhancedSemanticRole, SemanticComplex]]] = None,
		doc: Optional[spacy.tokens.Doc] = None
	) -> Dict[str, Any]:
		"""Format token analysis results for display, enriching with semantic role info."""
		enriched_matches = []
		
		if doc is None:
			logger.error("The 'doc' object is None in _format_analysis_for_display.")
			return {"token_matches": enriched_matches}
		
		for match in matches:
			match_info = {
				"Original Text": match.original_text,
				"Source": match.source,
				"Type": match.label or match.category,
				"Start": match.start,
				"End": match.end,
				"POS": match.pos_tag,
				"Lemma": match.lemma,
				"Semantic Role": "NONE"
			}
			
			if semantic_roles:
				match_span = doc.char_span(match.start, match.end, alignment_mode="expand")
				if match_span:
					for item in semantic_roles:
						if isinstance(item, SemanticComplex):
							for role in [item.base] + item.modifiers + item.scope:
								if self._check_span_overlap(match_span, role):
									match_info["Semantic Role"] = f"{role.role} ({role.predicate})"
									break  # Stop after finding the first overlapping role
						elif isinstance(item, EnhancedSemanticRole):
							if self._check_span_overlap(match_span, item):
								match_info["Semantic Role"] = f"{item.role} ({item.predicate})"
								break  # Stop after finding the first overlapping role
							
			enriched_matches.append(match_info)
			
		return {"token_matches": enriched_matches}
	
	def _check_span_overlap(self, span: spacy.tokens.Span, role: EnhancedSemanticRole) -> bool:
		"""Check if a spaCy Span overlaps with a semantic role's span."""
		role_span = role.token.doc.char_span(role.start, role.end, alignment_mode="expand")
		if role_span:
			# Check if span is completely within role_span
			return role_span.start <= span.start < span.end <= role_span.end
		return False
	
	def _check_token_overlap(self, token: Token, role: Union[EnhancedSemanticRole, SemanticComplex]) -> bool:
		"""Check if token overlaps with semantic role, excluding predicate tokens."""
		if isinstance(role, SemanticComplex):
			# Check overlap with any part of the complex
			return any(self._check_token_overlap(token, r) for r in [role.base] + role.modifiers + role.scope)
		else:
			# Exclude overlap check if the token is part of the predicate
			if token.start >= role.start and token.end <= role.end:
				if role.token is not None:
					doc = role.token.doc  # Access the doc from the token
					predicate_token = doc.char_span(role.start, role.end, alignment_mode="expand")
					if predicate_token:
						if any(token.start == t.idx and token.end == t.idx + len(t.text) for t in predicate_token):
							return False
				return True
			return False
		
	def process_and_store_sentence_matches(
		self,
		dialogue,
		sentence_idx,
		sentence,
		entities,
		mwes,
		regex_patterns,
		custom_stopwords,
		language_automaton, # Pass language automaton separately
		options, # Use a dedicated options dictionary
	):
		"""Process matches for a single sentence and store results."""
		# Get unified matcher results
		matches = unified_match(
			sentence,
			entities=entities,
			mwes=mwes,
			regex_patterns=regex_patterns,
			custom_stopwords=custom_stopwords,
			include_stopwords=options["include_stopwords"],
			min_noun_chunk_length=options["min_noun_chunk_length"],
			language_automaton=language_automaton,
			expand_trailing=False, # You might want to add this as an option
		)
		
		# Access options from the dictionary
		clean_text = options["clean_text"]
		use_lemma = options.get("use_lemma", False)  # Use .get() with defaults
		lowercase = options.get("lowercase", False)
		remove_stopwords = options.get("remove_stopwords", True)
		filter_type = options["filter_type"]
		
		# Filter matches
		filtered_matches = [
			match
			for match in matches
			if filter_type.lower() == "all"
			or match.source.lower() == filter_type.lower()
		]
		
		# Get semantic roles if semantic handler is available
		semantic_roles = None
		doc = None
		if hasattr(self, "semantic_handler"):
			doc = self.semantic_handler.nlp(sentence)
			semantic_roles = self.semantic_handler.extract_roles_from_doc(doc, filtered_matches) # Pass the matches
			
			# Store semantic analysis separately
			if not hasattr(dialogue, "semantic_outputs"):
				dialogue.semantic_outputs = {}
			dialogue.semantic_outputs[sentence_idx] = {
				"roles": semantic_roles,
			}
			
		# Clean text if enabled
		if clean_text:
			cleaned_sentence = clean_matched_text(
				sentence,
				filtered_matches,
				use_lemma=use_lemma,
				lowercase=lowercase,
				remove_stopwords=remove_stopwords,
			)
			if not hasattr(dialogue, "cleaned_sentences"):
				dialogue.cleaned_sentences = {}
			dialogue.cleaned_sentences[sentence_idx] = cleaned_sentence
			
		# Format and store token analysis results
		analysis_results = self._format_analysis_for_display(
			filtered_matches, semantic_roles, doc  # Pass doc here
		)
		
		# Store token matches
		if not hasattr(dialogue, "sentence_outputs"):
			dialogue.sentence_outputs = {}
		dialogue.sentence_outputs[sentence_idx] = analysis_results["token_matches"]
		
	def process_sections(self, sections, entities, mwes, regex_patterns, custom_stopwords, language_automaton,
		options):
		"""Recursively process sections for dialogues and subsections."""
		for section in sections:
			for dialogue in section.dialogues:
				for idx, sentence in enumerate(dialogue.sentences):
					self.process_and_store_sentence_matches(
						dialogue,
						idx,
						sentence,
						entities,
						mwes,
						regex_patterns,
						custom_stopwords,
						language_automaton,
						options,  # Pass options
					)
				if hasattr(dialogue, "sentence_outputs"):
					for sentence_idx, outputs in dialogue.sentence_outputs.items():
						pass
						
			self.process_sections(section.subsections, entities, mwes, regex_patterns, custom_stopwords, language_automaton, options) # Pass all arguments here as well
			
	def filter_results(self, transcripts, filter_type):
		"""Filter stored results without reprocessing."""
		for transcript in transcripts:
			for section in transcript.sections:
				for dialogue in section.dialogues:
					if hasattr(dialogue, "sentence_outputs"):
						# Store original outputs to avoid filtering the source data
						if not hasattr(dialogue, "original_outputs"):
							dialogue.original_outputs = {
								idx: list(outputs)
								for idx, outputs in dialogue.sentence_outputs.items()
							}
						# Reset from original data before filtering
						dialogue.sentence_outputs = {
							idx: [
								match
								for match in dialogue.original_outputs[idx]
								if filter_type.lower() == "all"
								or match["Source"].lower() == filter_type.lower()
							]
							for idx in dialogue.original_outputs
						}
		return transcripts
	
	def setup_sidebar(self):
		"""Set up Streamlit sidebar options."""
		st.sidebar.header("Configuration")
		filter_type = st.sidebar.selectbox(
			"Filter by Source",
			options=["All", "Entity", "MWE", "Regex", "SpaCy", "Custom"],
			help="Filter matches by their source",
			key="filter_source",  # Add stable key to prevent reruns
		)
		include_stopwords = st.sidebar.checkbox("Include SpaCy Stopwords", value=False)
		min_noun_chunk_length = st.sidebar.slider(
			"Minimum Noun Chunk Length", min_value=1, max_value=10, value=1
		)
		
		# Add SpaCy model selection
		st.sidebar.header("Model Configuration")
		model_choice = st.sidebar.selectbox(
			"SpaCy Model",
			options=["Small (Default)", "Medium", "Large", "Transformer"],
			index=0,
			help="Larger models are more accurate but slower. Transformer requires GPU.",
		)
		model_map = {
			"Small (Default)": SPACY_MODEL_SM,
			"Medium": SPACY_MODEL_MD,
			"Large": SPACY_MODEL_LG,
			"Transformer": SPACY_MODEL_TRF,
		}
		
		if model_choice != "Small (Default)":
			try:
				model_name = model_map[model_choice]
				set_spacy_model(model_name)
				if SPACY_MODEL_INFO[model_name]["gpu_required"]:
					st.warning("⚠️ Transformer model works best with GPU acceleration")
			except OSError:
				st.error(f"Model {model_choice} not installed. Using default model.")
				
		clean_text = st.sidebar.checkbox("Enable Text Cleaning", value=False)
		
		text_cleaning_options = {}
		if clean_text:
			text_cleaning_options = {
				"use_lemma": st.sidebar.checkbox("Lemmatize Tokens", value=True),
				"lowercase": st.sidebar.checkbox("Convert to Lowercase", value=True),
				"remove_stopwords": st.sidebar.checkbox("Remove Stopwords", value=True),
			}
			
		options = {
			"filter_type": filter_type,
			"include_stopwords": include_stopwords,
			"min_noun_chunk_length": min_noun_chunk_length,
			"clean_text": clean_text,
			**text_cleaning_options,
		}
		
		return options
	
		return {
			"filter_type": filter_type,
			"include_stopwords": include_stopwords,
			"min_noun_chunk_length": min_noun_chunk_length,
			"clean_text": clean_text,
			**text_cleaning_options,
		}
	
	def match_tokens_in_transcripts(self):
		"""Parse and match tokens in transcripts using custom configurations."""
		files_to_process = upload_transcript_files()
		parsed_transcripts = self.transcript_handler.parse_files(files_to_process)
		entities = self.configs.get("entities", {})
		mwes = self.configs.get("multi_word_expressions", {})
		regex_patterns = self.configs.get("regex_patterns", {})
		custom_stopwords = self.configs.get("stopwords", set())
		
		sidebar_options = self.setup_sidebar()
		
		# Initialize session state if not already done
		if "transcripts" not in st.session_state:
			st.session_state.transcripts = None
			st.session_state.processed = False
			
		if st.sidebar.button("Process Transcript", disabled=not files_to_process):
			self.unfiltered_transcripts = []
			for transcript in parsed_transcripts:
				self.process_sections(
					transcript.sections,
					entities=entities,
					mwes=mwes,
					regex_patterns=regex_patterns,
					custom_stopwords=custom_stopwords,
					language_automaton=self.language_automaton, # Pass language_automaton separately
					options=sidebar_options,  # Pass the options dictionary
				)
				self.unfiltered_transcripts.append(transcript)
			self.matches_processed = True
			# Store in session state
			st.session_state.transcripts = self.unfiltered_transcripts
			st.session_state.processed = True
			
		# Use stored results if available, otherwise use instance results
		stored_transcripts = (
			st.session_state.transcripts
			if st.session_state.processed
			else self.unfiltered_transcripts
			if self.matches_processed
			else None
		)
		
		if stored_transcripts:
			filtered = self.filter_results(
				stored_transcripts, sidebar_options["filter_type"]
			)
			transcript_data = [
				(transcript, f"Transcript {i + 1}")
				for i, transcript in enumerate(filtered)
			]
			# Pass the nlp model to display_transcripts
			display_transcripts(transcript_data, self.semantic_handler.nlp)
			
			
if __name__ == "__main__":
	test_wrapper = TranscriptMatchingTestWrapper()
	test_wrapper.match_tokens_in_transcripts()
	