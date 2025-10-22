#!/usr/bin/env python3

# test_EntityMWEHandler.py

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import json
import tempfile
import os
import nltk
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from custom_file_utils import (
	load_custom_entities, 
	load_multi_word_entries, 
	load_regex_patterns,
	load_json_file,
	load_language_data,
	load_custom_stopwords,
	load_aspect_configuration
)
from transcript_handler import process_transcript
from transcript_structures import TranscriptData
from EntityMWEHandler import EntityMWEHandler
from spacy_handler import SpaCyHandler, POSToken, NounChunk
from semantic_role_handler import SemanticRoleHandler
from aspect_handler import AspectHandler
from data_types import EntityToken, ProcessedToken, SemanticRole, AspectTerm
from nltk.tokenize import sent_tokenize
import spacy

def set_page_config():
	st.set_page_config(
		page_title="EntityMWEHandler Test Module",
		layout="wide",
		initial_sidebar_state="expanded"
	)
	
	# Additional styling for UI elements
	st.markdown("""
		<style>
			.stSlider > div {
				height: 3rem;
			}
			.stSelectbox > div > div {
				height: 3rem;
			}
		</style>
		""", unsafe_allow_html=True)
	
# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def load_and_display_config():
	"""Load and display preprocessing configuration options."""
	CONFIG_PATH = "preprocessing_config.json"
	try:
		with open(CONFIG_PATH, 'r') as f:
			config = json.load(f)
	except FileNotFoundError:
		st.sidebar.error(f"Configuration file not found at {CONFIG_PATH}")
		return None
	
	st.sidebar.markdown("### Preprocessing Settings")
	
	# Create dictionary to store updated values
	updated_config = {}
	
	# Processing Configuration in an expander
	with st.sidebar.expander("Processing Configuration", expanded=False):
		for key, value in config.items():
			if isinstance(value, bool):
				updated_config[key] = st.checkbox(
					key, 
					value=value,
					help=f"Current setting: {value}"
				)
			elif key == "language":
				updated_config[key] = st.selectbox(
					key,
					options=["english", "indonesian"],
					index=0 if value == "english" else 1,
					help=f"Current setting: {value}"
				)
				
		st.markdown("### SpaCy Model")
		updated_config["spacy_model"] = st.selectbox(
			"SpaCy Model",
			["en_core_web_sm", "en_core_web_trf"],
			index=0,
			help="Select SpaCy model (transformer model provides better accuracy but requires separate installation)"
		)
		
		if st.button("Save Configuration"):
			try:
				with open(CONFIG_PATH, 'w') as f:
					json.dump(updated_config, f, indent=2)
				st.success("Configuration saved successfully!")
			except Exception as e:
				st.error(f"Error saving configuration: {str(e)}")
				
				
				
	# Parallel Processing Settings in an expander
	with st.sidebar.expander("Parallel Processing Settings", expanded=False):
		updated_config["max_workers"] = st.slider("Max Workers", 1, 8, 4)
		updated_config["batch_size"] = st.slider("Batch Size", 1, 50, 10)
		updated_config["chunk_size"] = st.slider("Chunk Size", 100, 5000, 1000)
		
	return updated_config

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

def format_text_with_linebreaks(text: str, max_line_length: int = 80) -> str:
	"""Format text with appropriate line breaks for display."""
	words = text.split()
	lines = []
	current_line = []
	
	for word in words:
		if len(' '.join(current_line + [word])) > max_line_length:
			lines.append(' '.join(current_line))
			current_line = [word]
		else:
			current_line.append(word)
			
	if current_line:
		lines.append(' '.join(current_line))
		
	return '\n'.join(lines)

def mark_identified_tokens(text: str, identified_tokens: List[Union[Tuple, EntityToken]]) -> str:
	"""Mark identified entities in text for display."""
	# Handle both legacy tuples and EntityToken objects during transition
	def get_token_info(token):
		if isinstance(token, tuple):
			return token[3], token[4]  # start, end from tuple
		return token.start, token.end  # start, end from EntityToken
	
	sorted_tokens = sorted(identified_tokens, key=lambda x: get_token_info(x)[0])
	offset = 0
	result = text
	
	for token in sorted_tokens:
		start, end = get_token_info(token)
		adj_start = start + offset
		adj_end = end + offset
		result = (
			result[:adj_start] +
			'[' +
			result[adj_start:adj_end] +
			']' +
			result[adj_end:]
		)
		offset += 2
		
	return result

def mark_stopwords(text: str, processed_tokens: List[Union[Tuple, ProcessedToken]]) -> str:
	"""Mark stopwords in text for display."""
	def get_token_info(token):
		if isinstance(token, tuple):
			return token[0], token[3], token[4]  # text, action, source from tuple
		return token.text, token.action, token.source
	
	marked_text = text
	offset = 0
	
	for token in reversed(processed_tokens):
		text, action, source = get_token_info(token)
		start = marked_text.lower().rfind(text.lower(), 0, len(marked_text) - offset)
		if start != -1:
			end = start + len(text)
			if source.startswith('Protected'):
				marked_text = f"{marked_text[:start]}[{text}]{marked_text[end:]}"
			elif action == 'Stopword':
				marked_text = f"{marked_text[:start]}({text}){marked_text[end:]}"
			offset = len(marked_text) - end
	return marked_text

def mark_spans_in_text(text: str, spans: List[Tuple[int, int, str]]) -> str:
	"""Mark multiple span types in text with different markers."""
	chars = list(text)
	markers = []
	
	bracket_map = {
		'pos': ('⟦', '⟧'),
		'chunk': ('⟨', '⟩'),
		'entity': ('[', ']'),
		'semantic': ('{', '}'),
		'aspect': ('「', '」')
	}
	
	# Sort spans to handle overlaps and prioritize semantic roles
	# Semantic roles should appear outermost when overlapping with other spans
	spans = sorted(spans, key=lambda x: (
		x[0],  # Start position
		-len(x[2]),  # Longer span types first
		0 if x[2] == 'semantic' else 1  # Prioritize semantic spans
	))
	
	for start, end, span_type in spans:
		open_bracket, close_bracket = bracket_map.get(span_type, ('[', ']'))
		markers.append((start, open_bracket, 0))
		markers.append((end, close_bracket, 1))
		
	markers.sort(key=lambda x: (x[0], x[2]))
	offset = 0
	
	for pos, marker, _ in markers:
		chars.insert(pos + offset, marker)
		offset += 1
		
	return ''.join(chars)

def create_match_dataframe(matches: List[Dict]) -> pd.DataFrame:
	"""Create DataFrame for lexical matches."""
	if matches:
		df = pd.DataFrame([
			{
				'Term': match['term'],
				'Type': match['type'].title(),
				'POS': match.get('pos_tag', match.get('pos', 'Unknown')),  # Handle both possible keys
				'Match Type': match.get('match_type', 'word'),
				'Entity': match['entity_type'] or 'None',
				'Position': f"{match['position'][0]}-{match['position'][1]}"
			}
			for match in matches
		])
		return df.sort_values('Position')
	return pd.DataFrame()

def show_matches_in_context(matches: List[Dict], text: str):
	"""Show lexical matches in text context."""
	if not matches:
		return
	
	st.markdown("*In Context:*")
	marked_text = text
	offset = 0
	
	for match in sorted(matches, key=lambda x: x['position'][0]):
		start, end = match['position']
		start += offset
		end += offset
		
		annotation = f"{match.get('pos_tag', match.get('pos', 'Unknown'))}"  # Handle both possible keys
		if match.get('match_type'):
			annotation += f"|{match['match_type']}"
		if match['entity_type']:
			annotation += f"|{match['entity_type']}"
			
		marked_text = (
			marked_text[:start] +
			f"[{marked_text[start:end]}({annotation})]" +
			marked_text[end:]
		)
		offset += len(annotation) + 3
		
	st.text(marked_text)
	
def add_pos_distribution_analysis(lexical_features: Dict[str, List[Dict]]):
	"""Display POS distribution across lexical categories."""
	st.markdown("### POS Tag Distribution")
	
	pos_stats = {}
	for category, matches in lexical_features.items():
		if matches:
			pos_counts = {}
			for match in matches:
				pos_tag = match.get('pos_tag', match.get('pos', 'Unknown'))  # Handle both possible keys
				pos_counts[pos_tag] = pos_counts.get(pos_tag, 0) + 1
			pos_stats[category] = pos_counts
			
	if pos_stats:
		pos_data = []
		for category, pos_counts in pos_stats.items():
			total = sum(pos_counts.values())
			for pos, count in pos_counts.items():
				pos_data.append({
					'Category': category.title(),
					'POS': pos,
					'Count': count,
					'Percentage': round(count/total * 100, 1)
				})
				
		pos_df = pd.DataFrame(pos_data)
		st.dataframe(
			pos_df.pivot(index='Category', columns='POS', values='Count').fillna(0),
			use_container_width=True
		)
		
def add_match_type_analysis(lexical_features: Dict[str, List[Dict]]):
	"""Display analysis of match types across categories."""
	st.markdown("### Match Type Analysis")
	
	match_stats = []
	for category, matches in lexical_features.items():
		if matches:
			type_counts = {
				'word': sum(1 for m in matches if m.get('match_type', 'word') == 'word'),
				'phrase': sum(1 for m in matches if m.get('match_type') == 'phrase')
			}
			total = len(matches)
			match_stats.append({
				'Category': category.title(),
				'Words': type_counts['word'],
				'Phrases': type_counts['phrase'],
				'Word %': round(type_counts['word']/total * 100, 1) if total > 0 else 0,
				'Phrase %': round(type_counts['phrase']/total * 100, 1) if total > 0 else 0
			})
			
	if match_stats:
		st.dataframe(pd.DataFrame(match_stats), use_container_width=True)
		
def display_lexical_matches(text: str, analysis_results: Dict[str, Any]):
	"""Display detailed lexical category matches."""
	st.markdown("### Lexical Category Matches")
	
	lexical_features = analysis_results.get('lexical_features', {})
	
	col1, col2, col3 = st.columns(3)
	
	with col1:
		st.markdown("**Modifiers**")
		modifiers = lexical_features.get('modifiers', [])
		if modifiers:
			df = create_match_dataframe(modifiers)
			st.dataframe(df, use_container_width=True)
			show_matches_in_context(modifiers, text)
		else:
			st.info("No modifiers found")
			
	with col2:
		st.markdown("**Semantic Indicators**")
		semantic_matches = []
		for category in ['uncertainty', 'negation', 'causal']:
			semantic_matches.extend(lexical_features.get(category, []))
			
		if semantic_matches:
			df = create_match_dataframe(semantic_matches)
			st.dataframe(df, use_container_width=True)
			show_matches_in_context(semantic_matches, text)
		else:
			st.info("No semantic indicators found")
			
	with col3:
		st.markdown("**Intensity and Tone**")
		tone_matches = []
		for category in ['intensifiers', 'sarcasm']:
			tone_matches.extend(lexical_features.get(category, []))
			
		if tone_matches:
			df = create_match_dataframe(tone_matches)
			st.dataframe(df, use_container_width=True)
			show_matches_in_context(tone_matches, text)
		else:
			st.info("No intensity or tone markers found")
			
	# Statistics Section
	stats_data = []
	total_matches = 0
	for category, matches in lexical_features.items():
		if matches:
			category_count = len(matches)
			total_matches += category_count
			unique_terms = len(set(m['term'].lower() for m in matches))
			entity_overlap = sum(1 for m in matches if m['entity_type'])
			phrase_count = sum(1 for m in matches if m.get('match_type') == 'phrase')
			
			stats_data.append({
				'Category': category.title(),
				'Total Matches': category_count,
				'Unique Terms': unique_terms,
				'Phrases': phrase_count,
				'Entity Overlaps': entity_overlap,
			})
			
	if stats_data:
		st.markdown("### Lexical Analysis Details")
		st.markdown("#### Category Statistics")
		stats_df = pd.DataFrame(stats_data)
		st.dataframe(stats_df, use_container_width=True)
		
		add_pos_distribution_analysis(lexical_features)
		add_match_type_analysis(lexical_features)
		
		# Summary metrics
		cols = st.columns(5)
		with cols[0]:
			st.metric("Total Matches", total_matches)
		with cols[1]:
			unique_total = sum(row['Unique Terms'] for row in stats_data)
			st.metric("Unique Terms", unique_total)
		with cols[2]:
			phrase_total = sum(row['Phrases'] for row in stats_data)
			st.metric("Phrases", phrase_total)
		with cols[3]:
			entity_total = sum(row['Entity Overlaps'] for row in stats_data)
			st.metric("Entity Overlaps", entity_total)
		with cols[4]:
			categories_with_matches = sum(1 for row in stats_data if row['Total Matches'] > 0)
			st.metric("Active Categories", categories_with_matches)
			
def display_preprocessing_results(text: str, handler: EntityMWEHandler):
	"""Display enhanced preprocessing results with linguistic analysis."""
	processed_results = handler.process_text(text)[0]
	original_text = processed_results[0].split(": ", 1)[1] if ": " in processed_results[0] else processed_results[0]
	analysis_data = processed_results[1]
	
	st.markdown("**Original text:**")
	st.write(format_text_with_linebreaks(original_text))
	
	# Create spans for visualization
	spans = []
	if analysis_data.get('pos_tags'):
		spans.extend([(token.start, token.end, 'pos') for token in analysis_data['pos_tags']])
	if analysis_data.get('noun_chunks'):
		spans.extend([(chunk.start, chunk.end, 'chunk') for chunk in analysis_data['noun_chunks']])
	if analysis_data.get('aspects'):  # Add aspects spans
		spans.extend([
			(aspect.position[0], aspect.position[1], 'aspect') 
			for aspect in analysis_data['aspects']
		])
	spans.extend([
		(getattr(ent, 'start', ent[3]), getattr(ent, 'end', ent[4]), 'entity') 
		for ent in analysis_data['identified_tokens']
	])
	if analysis_data.get('semantic_roles'):  # New addition
		spans.extend([
			(role.start, role.end, 'semantic') 
			for role in analysis_data['semantic_roles']
		])
	
	st.markdown("### Text with Annotations")
	st.markdown("""
	- ⟦POS Tags⟧
	- ⟨Noun Chunks⟩
	- [Named Entities]
	- {Semantic Roles}
	- 「Aspects」
	""")
	
	marked_text = mark_spans_in_text(text, spans)
	st.text(marked_text)
	
	# Reordered tabs with Lexical Analysis after Cleaned Text
	tabs = st.tabs([
		"Entities", 
		"Stopwords", 
		"Cleaned Text", 
		"Semantic Roles",
		"Aspects",
		"Lexical Analysis", 
		"POS Analysis", 
		"Noun Chunks", 
		"spaCy Entities", 
		"Summary"
	])
	
	# Rest of the function remains the same, just need to reorder the tab content handlers
	with tabs[0]:  # Entities
		if analysis_data['identified_tokens']:
			st.markdown("**Text with marked entities:**")
			marked_text = mark_identified_tokens(original_text, analysis_data['identified_tokens'])
			st.write(format_text_with_linebreaks(marked_text))
			
			st.markdown("**Identified Entities:**")
			entities_df = pd.DataFrame([
				{
					'Entity': getattr(entity, 'text', entity[0]),
					'Type': getattr(entity, 'label', entity[1]),
					'Source': getattr(entity, 'source', entity[2]),
					'Position': f"{getattr(entity, 'start', entity[3])}-{getattr(entity, 'end', entity[4])}"
				}
				for entity in analysis_data['identified_tokens']
			])
			st.dataframe(entities_df)
			
	with tabs[1]:  # Stopwords
		st.markdown("**Stopwords Analysis:**")
		marked_stopwords = mark_stopwords(original_text, analysis_data['processed_tokens'])
		st.write(format_text_with_linebreaks(marked_stopwords))
		stopwords = [token for token in analysis_data['processed_tokens'] 
					if getattr(token, 'action', token[3]) == 'Stopword']
		if stopwords:
			stopwords_df = pd.DataFrame([
				{
					'Token': getattr(token, 'text', token[0]),
					'Lemma': getattr(token, 'lemma', token[1]),
					'POS': getattr(token, 'pos', token[2]),
					'Source': getattr(token, 'source', token[4])
				}
				for token in stopwords
			])
			st.dataframe(stopwords_df)
			
	with tabs[2]:  # Cleaned Text
		st.markdown("**Cleaned text:**")
		st.write(format_text_with_linebreaks(analysis_data['cleaned_text']))
	
	# Add new semantic roles tab handler
	with tabs[3]:  # Semantic Roles tab
		st.markdown("### Semantic Role Analysis")
		if analysis_data.get('semantic_roles'):
			# Create predicates section
			st.markdown("#### Predicate-Argument Structure")
			for predicate, arguments in analysis_data['predicate_arguments'].items():
				with st.expander(f"Predicate: {predicate}"):
					args_df = pd.DataFrame([
						{
							'Argument': arg['argument'],
							'Role': arg['role'],
							'Position': f"{arg['position'][0]}-{arg['position'][1]}"
						}
						for arg in arguments
					])
					st.dataframe(args_df)
					
			# Create roles section
			st.markdown("#### All Semantic Roles")
			roles_df = pd.DataFrame([
				{
					'Predicate': role.predicate,
					'Argument': role.argument,
					'Role': role.role,
					'Position': f"{role.start}-{role.end}"
				}
				for role in analysis_data['semantic_roles']
			])
			st.dataframe(roles_df)
		else:
			st.info("No semantic roles found in this text.")
			
	# Add new tab handler after existing tabs
	with tabs[4]:  # Aspects tab
		st.markdown("### Aspect Analysis")
		if analysis_data.get('aspects'):
			aspects_df = pd.DataFrame([
				{
					'Text': aspect.text,
					'Category': aspect.category,
					'Target': aspect.target or 'N/A',
					'Position': f"{aspect.position[0]}-{aspect.position[1]}",
					'Confidence': f"{aspect.confidence:.2f}"
				}
				for aspect in analysis_data['aspects']
			])
			st.dataframe(aspects_df)
			
			# Show aspects in context
			st.markdown("**Text with marked aspects:**")
			aspect_spans = [(a.position[0], a.position[1], 'aspect') 
							for a in analysis_data['aspects']]
			marked_text = mark_spans_in_text(text, aspect_spans)
			st.text(marked_text)
		else:
			st.info("No aspects found in this text.")
			
	with tabs[5]:  # Lexical Analysis
		if analysis_data.get('lexical_features'):
			display_lexical_matches(text, analysis_data)
		else:
			st.info("No lexical features found")
			
	with tabs[6]:  # POS Analysis
		st.markdown("### Part of Speech Analysis")
		if analysis_data.get('pos_tags'):
			df = pd.DataFrame([
				{
					'Text': token.text,
					'Lemma': token.lemma,
					'POS': token.pos,
					'Detailed Tag': token.tag,
					'Dependency': token.dep,
					'Position': f"{token.start}-{token.end}"
				}
				for token in analysis_data['pos_tags']
			])
			st.dataframe(df)
		else:
			st.info("No relevant POS tokens found.")
			
	with tabs[7]:  # Noun Chunks
		st.markdown("### Noun Chunks")
		if analysis_data.get('noun_chunks'):
			df = pd.DataFrame([
				{
					'Chunk': chunk.text,
					'Root': chunk.root_text,
					'Root Dependency': chunk.root_dep,
					'Position': f"{chunk.start}-{chunk.end}"
				}
				for chunk in analysis_data['noun_chunks']
			])
			st.dataframe(df)
		else:
			st.info("No noun chunks found.")
			
	with tabs[8]:  # spaCy Entities
		st.markdown("### Named Entities (spaCy)")
		if analysis_data.get('spacy_entities'):
			entities_df = pd.DataFrame([
				{
					'Entity': entity[0],
					'Type': entity[1],
					'Position': f"{entity[3]}-{entity[4]}"
				}
				for entity in analysis_data['spacy_entities']
			])
			st.dataframe(entities_df)
			
			# Optionally show entities in context
			st.markdown("**Text with spaCy entities:**")
			spans = [(ent[3], ent[4], 'entity') for ent in analysis_data['spacy_entities']]
			marked_text = mark_spans_in_text(original_text, spans)
			st.text(marked_text)
			
		else:
			st.info("No entities found.")    
			
	with tabs[9]:  # Summary
		st.markdown("### Analysis Summary")
		cols = st.columns(6)
		with cols[0]:
			st.metric("POS Tags", len(analysis_data.get('pos_tags', [])))
		with cols[1]:
			st.metric("Noun Chunks", len(analysis_data.get('noun_chunks', [])))
		with cols[2]:
			st.metric("Entities", len(analysis_data['identified_tokens']))
		with cols[3]:
			st.metric("Semantic Roles", len(analysis_data.get('semantic_roles', [])))
		with cols[4]:
			st.metric("Aspects", len(analysis_data.get('aspects', [])))
		with cols[5]:
			total_lexical = sum(
				len(matches) 
				for matches in analysis_data.get('lexical_features', {}).values()
			)
			st.metric("Lexical Matches", total_lexical)
			
def process_transcript_section(handler: EntityMWEHandler, sections: List[Any]):
	"""Process transcript sections using tabbed layout."""
	if not sections:
		return
	
	# Create tabs for each section
	section_tabs = st.tabs([section.name for section in sections])
	
	for section, section_tab in zip(sections, section_tabs):
		with section_tab:
			if section.dialogues:
				# Create row of dialogue tabs
				dialogue_tabs = st.tabs([f"Dialogue {i+1}: {dialogue.speaker}" 
										for i, dialogue in enumerate(section.dialogues)])
				
				for dialogue, dialogue_tab in zip(section.dialogues, dialogue_tabs):
					with dialogue_tab:
						st.markdown(f"**Speaker:** {dialogue.speaker}")
						st.markdown(f"**Role:** {dialogue.role}")
						
						# Get sentences and create sentence tabs
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
										display_preprocessing_results(sentences[sentence_idx], handler)
										
			# Process subsections recursively using the same tabbed layout
			if section.subsections:
				st.markdown("### Subsections")
				process_transcript_section(handler, section.subsections)
				
def main():
	st.title("EntityMWEHandler Preprocessing Test Module")
	
	# Load and display configuration
	config = load_and_display_config()
	if not config:
		st.stop()
		
	# File upload section
	st.header("Upload Files")
	
	with st.expander("Optional Preprocessing Files"):
		custom_entities_files = st.file_uploader(
			"Custom Entities JSON (Optional)", 
			type="json", 
			accept_multiple_files=True
		)
		mwe_files = st.file_uploader(
			"Multi-Word Expressions JSON (Optional)", 
			type="json", 
			accept_multiple_files=True
		)
		language_data_files = st.file_uploader(
			"Language Data JSON (Optional)", 
			type="json", 
			accept_multiple_files=True
		)
		regex_patterns_files = st.file_uploader(
			"Regex Patterns JSON (Optional)", 
			type="json", 
			accept_multiple_files=True
		)
		custom_stopwords_files = st.file_uploader(
			"Custom Stopwords JSON (Optional)", 
			type="json", 
			accept_multiple_files=True
		)
		aspect_config_files = st.file_uploader(  # Changed to support multiple files
			"Aspect Configuration JSON (Optional)", 
			type="json",
			accept_multiple_files=True
		)
		
	transcript_files = st.file_uploader(
		"Upload Transcript Files (Required)", 
		type=["json", "txt", "pdf"], 
		accept_multiple_files=True
	)
	
	if not transcript_files:
		st.warning("Please upload at least one transcript file to proceed.")
		return
	
	if st.button("Start Preprocessing", type="primary"):
		try:
			# Save uploaded files
			custom_entities_paths = save_uploaded_files(custom_entities_files)
			mwe_paths = save_uploaded_files(mwe_files)
			regex_patterns_paths = save_uploaded_files(regex_patterns_files)
			language_data_paths = save_uploaded_files(language_data_files)
			custom_stopwords_paths = save_uploaded_files(custom_stopwords_files)
			aspect_config_paths = save_uploaded_files(aspect_config_files)
			
			# Initialize handlers
			language_data = load_language_data(language_data_paths) if language_data_paths else {}
			spacy_handler = SpaCyHandler(language_data_paths if language_data_paths else {})
			
			handler = EntityMWEHandler(
				custom_entities_paths or [], 
				mwe_paths or [], 
				regex_patterns_paths or [], 
				language_data,
				spacy_handler,
				custom_stopwords_paths or [],
				aspect_config_files=aspect_config_paths or [],
				preprocessing_config=config
			)
			
			# Process transcripts
			for transcript_file in transcript_files:
				st.subheader(f"Processing: {transcript_file.name}")
				
				# Save and process transcript
				file_type = transcript_file.type.split('/')[1]
				with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
					temp_file.write(transcript_file.read())
					temp_file_path = temp_file.name
					
				transcript_data, _ = process_transcript(temp_file_path, file_type)
				
				# Process all sections at once with new tabbed layout
				process_transcript_section(handler, transcript_data.sections)
				
				os.unlink(temp_file_path)
				st.success(f"Completed preprocessing {transcript_file.name}")
				
		except Exception as e:
			st.error(f"An error occurred during preprocessing: {str(e)}")
			st.error(f"Error details: {type(e).__name__}, {str(e)}")
			
		finally:
			# Cleanup temporary files
			for path_list in [custom_entities_paths, mwe_paths, regex_patterns_paths,
							language_data_paths, custom_stopwords_paths, aspect_config_paths]:
				if path_list:
					for path in path_list:
						try:
							os.unlink(path)
						except Exception as e:
							st.warning(f"Could not remove temporary file {path}: {str(e)}")
							
if __name__ == "__main__":
	set_page_config()
	main()