#!/usr/bin/env python3

# test_spacy_pipeline.py

import streamlit as st
import json
import tempfile
import os
from typing import List, Tuple, Dict, Any
import pandas as pd
import spacy
import logging
from spacy_pipeline_handler import SpaCyPipelineHandler
import nltk
from nltk.tokenize import sent_tokenize
from data_types import SemanticRole, AspectTerm, EntityToken, ProcessedToken
from transcript_handler import process_transcript
from transcript_structures import TranscriptData
from custom_file_utils import (
	load_custom_entities,
	load_multi_word_entries,
	load_regex_patterns,
	load_language_data,
	load_custom_stopwords,
	load_aspect_configuration
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt', quiet=True)

def set_page_config():
	"""Configure Streamlit page settings."""
	st.set_page_config(
		page_title="spaCy Pipeline Tester",
		layout="wide",
		initial_sidebar_state="expanded"
	)
	
class SpacyPipelineTester:
	"""Test interface for spaCy-based NLP pipeline."""
	
	def __init__(self, 
				language_data_files: List[str] = None,
				custom_entities_files: List[str] = None,
				mwe_files: List[str] = None,
				regex_patterns_files: List[str] = None,
				custom_stopwords_files: List[str] = None,
				aspect_config_files: List[str] = None, 
				model: str = "en_core_web_sm"):
		
		# Initialize spaCy pipeline handler instead of individual components
		self.pipeline_handler = SpaCyPipelineHandler(
			language_data_files=language_data_files,
			custom_entities_files=custom_entities_files,
			mwe_files=mwe_files,
			regex_patterns_files=regex_patterns_files,
			custom_stopwords_files=custom_stopwords_files,
			model=model
		)
		
	def analyze_text(self, text: str) -> Dict[str, Any]:
		"""Delegate text analysis to pipeline handler."""
		try:
			results = self.pipeline_handler.analyze_text(text)
			
			return results
		except Exception as e:
			logger.error(f"Error in text analysis: {str(e)}")
			raise
			
def save_uploaded_files(uploaded_files: List[Any]) -> List[str]:
	"""Save uploaded files to temporary location."""
	if not uploaded_files:
		return []
	temp_file_paths = []
	for uploaded_file in uploaded_files:
		with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
			if isinstance(uploaded_file, bytes):
				temp_file.write(uploaded_file)
			else:
				temp_file.write(uploaded_file.read())
			temp_file_paths.append(temp_file.name)
	return temp_file_paths

def load_and_display_config():
	"""Load and display preprocessing configuration options."""
	CONFIG_PATH = "preprocessing_config.json"
	try:
		with open(CONFIG_PATH, 'r') as f:
			config = json.load(f)
	except FileNotFoundError:
		st.sidebar.error(f"Configuration file not found at {CONFIG_PATH}")
		return None
	
	st.sidebar.markdown("### Processing Settings")
	
	# Create dictionary to store updated values
	updated_config = {}
	
	# Processing Configuration
	with st.sidebar.expander("Core Configuration", expanded=False):
		for key, value in config.items():
			if isinstance(value, bool):
				updated_config[key] = st.checkbox(
					key, value=value,
					help=f"Current setting: {value}"
				)
			elif key == "language":
				updated_config[key] = st.selectbox(
					key,
					options=["english", "indonesian"],
					index=0 if value == "english" else 1,
					help=f"Current setting: {value}"
				)
				
	# Model Selection
	with st.sidebar.expander("Model Configuration", expanded=False):
		updated_config["spacy_model"] = st.selectbox(
			"SpaCy Model",
			["en_core_web_sm", "en_core_web_trf"],
			index=0,
			help="Select SpaCy model (transformer model requires separate installation)"
		)
		
	return updated_config

def process_transcript_section(pipeline: SpacyPipelineTester, sections: List[Any]):
	"""Process transcript sections using tabbed layout."""
	if not sections:
		return
	
	# Create tabs for each section
	section_tabs = st.tabs([section.name for section in sections])
	
	for section, section_tab in zip(sections, section_tabs):
		with section_tab:
			if section.dialogues:
				dialogue_tabs = st.tabs([f"Dialogue {i+1}: {dialogue.speaker}" 
										for i, dialogue in enumerate(section.dialogues)])
				
				for dialogue, dialogue_tab in zip(section.dialogues, dialogue_tabs):
					with dialogue_tab:
						st.markdown(f"**Speaker:** {dialogue.speaker}")
						st.markdown(f"**Role:** {dialogue.role}")
						
						# Get sentences and create sentence group tabs
						sentences = sent_tokenize(dialogue.text)
						SENTENCES_PER_GROUP = 10
						num_groups = (len(sentences) + SENTENCES_PER_GROUP - 1) // SENTENCES_PER_GROUP
						
						# Create group tabs first
						group_tabs = st.tabs([
							f"Sentences {i*SENTENCES_PER_GROUP + 1}-{min((i+1)*SENTENCES_PER_GROUP, len(sentences))}"
							for i in range(num_groups)
						])
						
						# Within each group tab, create sentence tabs
						for group_idx, group_tab in enumerate(group_tabs):
							with group_tab:
								start_idx = group_idx * SENTENCES_PER_GROUP
								end_idx = min((group_idx + 1) * SENTENCES_PER_GROUP, len(sentences))
								
								sentence_tabs = st.tabs([
									f"Sentence {i+1}" 
									for i in range(start_idx, end_idx)
								])
								
								for sentence_idx, tab in zip(range(start_idx, end_idx), sentence_tabs):
									with tab:
										results = pipeline.analyze_text(sentences[sentence_idx])
										display_analysis(sentences[sentence_idx], results)
										
			# Process subsections recursively
			if section.subsections:
				st.markdown("### Subsections")
				process_transcript_section(pipeline, section.subsections)
				
def mark_spans_in_text(text: str, spans: List[Tuple[int, int, str]]) -> str:
	"""
	Mark multiple span types in text with different markers.
	
	Args:
		text (str): Original text
		spans (List[Tuple[int, int, str]]): List of (start, end, type) tuples
		
	Returns:
		str: Text with marked spans using brackets
	"""
	chars = list(text)
	markers = []
	
	# Define different brackets for different span types
	bracket_map = {
		'entity': ('[', ']'),
		'mwe': ('⟨', '⟩'),  # Using angle brackets for MWEs
		'regex': ('⟦', '⟧'),  # Using double brackets for regex matches
		'stopword': ('《', '》')  # Using guillemets for stopwords
	}
	
	# Add markers for each span
	for start, end, span_type in spans:
		open_bracket, close_bracket = bracket_map.get(span_type, ('[', ']'))
		markers.append((start, open_bracket, 0))  # 0 for opening bracket
		markers.append((end, close_bracket, 1))   # 1 for closing bracket
		
	# Sort markers by position, with closing brackets coming after opening ones
	# at the same position
	markers.sort(key=lambda x: (x[0], x[2]))
	
	# Insert markers into text
	offset = 0
	for pos, marker, _ in markers:
		chars.insert(pos + offset, marker)
		offset += 1
		
	return ''.join(chars)

def mark_stopwords(text: str, stopword_tokens: List[Tuple[str, str, str, str, str]]) -> str:
	"""
	Mark stopwords in text.
	
	Args:
		text (str): Original text
		stopword_tokens: List of (text, lemma, pos, action, source) tuples
		
	Returns:
		str: Text with marked stopwords
	"""
	# Create spans for stopwords
	spans = []
	pos = 0
	
	for token_text, _, _, action, _ in stopword_tokens:
		if action == 'Stopword':
			# Find the token in text from current position
			token_pos = text.find(token_text, pos)
			if token_pos != -1:
				spans.append((token_pos, token_pos + len(token_text), 'stopword'))
				pos = token_pos + len(token_text)
				
	return mark_spans_in_text(text, spans)

def display_analysis(text: str, results: Dict[str, Any]):
	"""Display analysis results in Streamlit interface."""
	st.subheader("Analysis Results")
	
	tabs = st.tabs([
		"Overview", 
		"Linguistic Features",
		"Custom Entities",  # New
		"Multi-Word Expressions",  # New
		"Regex Patterns",  # New
		"Stopwords",  # New
		"Semantic Roles", 
		"Aspects"
	])
	
	with tabs[0]:
		# Show original text
		st.markdown("**Original Text:**")
		st.text(text)
		
		if 'protected_spans' in results:
			protected_text = text
			for start, end in sorted(results['protected_spans'], reverse=True):
				protected_text = (
					protected_text[:start] + 
					'[' + protected_text[start:end] + ']' + 
					protected_text[end:]
				)
			st.markdown("**Protected Spans:**")
			st.text(protected_text)
			
		# Statistics in columns
		st.markdown("### Statistics")
		col1, col2, col3 = st.columns(3)  # Changed to 3 columns
		with col1:
			st.metric("Entities", len(results.get('entities', [])))
			st.metric("POS Tags", len(results.get('pos_tags', [])))
		with col2:
			st.metric("Semantic Roles", len(results.get('semantic_roles', [])))
			st.metric("Aspects", len(results.get('aspects', [])))
		with col3:
			st.metric("Processed Tokens", len(results.get('processed_tokens', [])))
			st.metric("Protected Spans", len(results.get('protected_spans', [])))
			
		# Token information in expandable section
		if 'processed_tokens' in results:
			with st.expander("Show Token Processing Details", expanded=False):
				st.markdown("### Token Processing Results")
				# Group tokens by action
				action_groups = {'Keep': [], 'Stopword': []}
				for token in results['processed_tokens']:
					action_groups[token.action].append({
						'Text': token.text,
						'Lemma': token.lemma,
						'POS': token.pos,
						'Source': token.source,
						'Entity Type': token.entity_type or 'None'
					})
					
				# Display kept tokens
				if action_groups['Keep']:
					st.markdown("**Kept Tokens:**")
					st.dataframe(pd.DataFrame(action_groups['Keep']))
					
				# Display stopwords
				if action_groups['Stopword']:
					st.markdown("**Stopwords:**")
					st.dataframe(pd.DataFrame(action_groups['Stopword']))
					
	with tabs[1]:
		if results.get('pos_tags'):
			st.markdown("### POS Tags")
			pos_df = pd.DataFrame([
				{
					'Text': t.text,
					'POS': t.pos,
					'Tag': t.tag,
					'Dep': t.dep
				} for t in results['pos_tags']
			])
			st.dataframe(pos_df)
			
		if results.get('entities'):
			st.markdown("### Entities")
			ent_df = pd.DataFrame([
				{
					'Text': e[0],
					'Type': e[1],
					'Source': e[2]
				} for e in results['entities']
			])
			st.dataframe(ent_df)
			
	with tabs[2]:  # Custom Entities
		if results.get('custom_entities'):
			st.markdown("### Custom Named Entities")
			entities_df = pd.DataFrame([
				{
					'Text': e[0],
					'Type': e[1],
					'Position': f"{e[3]}-{e[4]}"
				} for e in results['custom_entities']
			])
			st.dataframe(entities_df)
			# Add visualization of spans like in test_EntityMWEHandler
			st.markdown("**Text with marked entities:**")
			spans = [(e[3], e[4], 'entity') for e in results['custom_entities']]
			marked_text = mark_spans_in_text(text, spans)
			st.text(marked_text)
		else:
			st.info("No custom entities found")
			
	with tabs[3]:  # MWE
		if results.get('mwe'):
			st.markdown("### Multi-Word Expressions")
			mwe_df = pd.DataFrame([
				{
					'Expression': e[0],
					'Category': e[1],
					'Position': f"{e[3]}-{e[4]}"
				} for e in results['mwe']
			])
			st.dataframe(mwe_df)
			st.markdown("**Text with marked MWEs:**")
			spans = [(e[3], e[4], 'mwe') for e in results['mwe']]
			marked_text = mark_spans_in_text(text, spans)
			st.text(marked_text)
		else:
			st.info("No multi-word expressions found")
			
	with tabs[4]:  # Regex
		if results.get('regex_matches'):
			st.markdown("### Regex Pattern Matches")
			regex_df = pd.DataFrame([
				{
					'Text': e[0],
					'Pattern': e[1],
					'Position': f"{e[3]}-{e[4]}"
				} for e in results['regex_matches']
			])
			st.dataframe(regex_df)
			st.markdown("**Text with marked patterns:**")
			spans = [(e[3], e[4], 'regex') for e in results['regex_matches']]
			marked_text = mark_spans_in_text(text, spans)
			st.text(marked_text)
		else:
			st.info("No regex patterns found")
			
	with tabs[5]:  # Stopwords
		if results.get('stopwords'):
			st.markdown("### Stopwords")
			stopwords_df = pd.DataFrame([
				{
					'Token': token[0],
					'Lemma': token[1],
					'POS': token[2],
					'Action': token[3],
					'Source': token[4]
				} for token in results['stopwords']
				if token[3] == 'Stopword'   # Only show actual stopwords
			])
			st.dataframe(stopwords_df)
			st.markdown("**Text with marked stopwords:**")
			marked_text = mark_stopwords(text, results['stopwords'])
			st.text(marked_text)
		else:
			st.info("No stopwords found")
			
	with tabs[6]:
		if results.get('semantic_roles'):
			st.markdown("### Semantic Roles")
			roles_df = pd.DataFrame([
				{
					'Predicate': role.predicate,
					'Argument': role.argument,
					'Role': role.role
				} for role in results['semantic_roles']
			])
			st.dataframe(roles_df)
			
	with tabs[7]:
		if results.get('aspects'):
			st.markdown("### Aspects")
			aspects_df = pd.DataFrame([
				{
					'Text': aspect.text,
					'Category': aspect.category,
					'Target': aspect.target or 'N/A',
					'Confidence': f"{aspect.confidence:.2f}"
				} for aspect in results['aspects']
			])
			st.dataframe(aspects_df)
			
def main():
	set_page_config()
	st.title("spaCy Pipeline Tester")
	
	# Configuration
	config = load_and_display_config()
	if not config:
		st.stop()
		
	# Initialize temp_files list
	temp_files = []
	
	# File uploads section
	st.header("Upload Files")
	
	with st.expander("Optional Processing Files", expanded=True):
		custom_entities_files = st.file_uploader(
			"Custom Entities JSON (Optional)",
			type="json",
			accept_multiple_files=True,
			help="Upload files containing custom named entity definitions"
		)
		
		mwe_files = st.file_uploader(
			"Multi-Word Expressions JSON (Optional)",
			type="json",
			accept_multiple_files=True,
			help="Upload files containing multi-word expression patterns"
		)
		
		regex_patterns_files = st.file_uploader(
			"Regex Patterns JSON (Optional)",
			type="json",
			accept_multiple_files=True,
			help="Upload files containing custom regex patterns"
		)
		
		custom_stopwords_files = st.file_uploader(
			"Custom Stopwords JSON (Optional)",
			type="json",
			accept_multiple_files=True,
			help="Upload files containing custom stopwords"
		)
		
		language_data_files = st.file_uploader(
			"Language Data JSON (Optional)",
			type="json",
			accept_multiple_files=True,
			help="Upload files containing language processing rules"
		)
		
		aspect_config_files = st.file_uploader(
			"Aspect Configuration JSON (Optional)",
			type="json",
			accept_multiple_files=True,
			help="Upload files containing aspect analysis configuration"
		)
		
	transcript_files = st.file_uploader(
		"Upload Transcript Files (Required)",
		type=["json", "txt", "pdf"],
		accept_multiple_files=True,
		help="Upload transcript files for processing"
	)
	
	try:
		# Save all uploaded files if provided
		custom_entities_paths = save_uploaded_files(custom_entities_files) if custom_entities_files else []
		mwe_paths = save_uploaded_files(mwe_files) if mwe_files else []
		regex_patterns_paths = save_uploaded_files(regex_patterns_files) if regex_patterns_files else []
		custom_stopwords_paths = save_uploaded_files(custom_stopwords_files) if custom_stopwords_files else []
		language_data_paths = save_uploaded_files(language_data_files) if language_data_files else []
		aspect_config_paths = save_uploaded_files(aspect_config_files) if aspect_config_files else []
		
		# Track all temp files
		temp_files.extend(custom_entities_paths)
		temp_files.extend(mwe_paths)
		temp_files.extend(regex_patterns_paths)
		temp_files.extend(custom_stopwords_paths)
		temp_files.extend(language_data_paths)
		temp_files.extend(aspect_config_paths)
		
		# Initialize pipeline
		pipeline = SpacyPipelineTester(
			language_data_files=language_data_paths[0] if language_data_paths else None,
			custom_entities_files=custom_entities_paths if custom_entities_paths else None,
			mwe_files=mwe_paths if mwe_paths else None,
			regex_patterns_files=regex_patterns_paths if regex_patterns_paths else None,
			custom_stopwords_files=custom_stopwords_paths if custom_stopwords_paths else None,
			aspect_config_files=aspect_config_paths[0] if aspect_config_paths else None,
			model=config.get('spacy_model', 'en_core_web_sm')
		)
		
		# Handle both direct text input and transcript processing
		text_input = st.text_area("Or enter text directly to analyze")
		
		if text_input and st.button("Analyze Text"):
			results = pipeline.analyze_text(text_input)
			display_analysis(text_input, results)
			
		elif transcript_files:
			if st.button("Process Transcripts", type="primary"):
				for transcript_file in transcript_files:
					st.subheader(f"Processing: {transcript_file.name}")
					
					# Save and process transcript
					file_type = transcript_file.type.split('/')[1]
					with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
						temp_file.write(transcript_file.read())
						temp_file_path = temp_file.name
						temp_files.append(temp_file_path)
						
						transcript_data, _ = process_transcript(temp_file_path, file_type)
						process_transcript_section(pipeline, transcript_data.sections)
						
					st.success(f"Completed processing {transcript_file.name}")
					
	except Exception as e:
		st.error(f"Error: {str(e)}")
		
	finally:
		# Cleanup temporary files
		for temp_file in temp_files:
			try:
				os.unlink(temp_file)
			except Exception as e:
				logger.warning(f"Could not remove temporary file {temp_file}: {e}")
				
if __name__ == "__main__":
	main()