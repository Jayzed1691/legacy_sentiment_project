#!/usr/bin/env python3

# custom_file_utils.py

import json
import logging
from typing import Dict, Any, List, Union
from streamlit.runtime.uploaded_file_manager import UploadedFile

logger = logging.getLogger(__name__)

def load_json_file(file: Union[str, UploadedFile]) -> Dict[str, Any]:
	"""
	Load a JSON file, either from a file path or an UploadedFile object.
	
	Args:
		file (Union[str, UploadedFile]): The file to load, either a file path or an UploadedFile object.
	
	Returns:
		Dict[str, Any]: The loaded JSON data.
	"""
	try:
		if isinstance(file, str):
			with open(file, 'r') as f:
				return json.load(f)
		elif isinstance(file, UploadedFile):
			return json.loads(file.read().decode())
		else:
			raise ValueError("Input must be either a file path or an UploadedFile object")
	except json.JSONDecodeError:
		logger.error(f"Error decoding JSON from {file}")
		return {}
	except Exception as e:
		logger.error(f"Error loading file: {str(e)}")
		return {}
	
def load_custom_stopwords(files: Union[str, UploadedFile, List[UploadedFile]]) -> Dict[str, List[str]]:
	"""
	Load custom stopwords from one or more JSON files.

	Args:
		files (Union[str, UploadedFile, List[UploadedFile]]): The file(s) to load stopwords from.

	Returns:
		Dict[str, List[str]]: A dictionary with a "stopwords" key containing the deduplicated list of stopwords.
	"""
	if isinstance(files, (str, UploadedFile)):
		files = [files]
		
	all_stopwords = []
	for file in files:
		try:
			content = load_json_file(file)
			if isinstance(content, dict) and "custom_stopwords" in content:
				all_stopwords.extend(content["custom_stopwords"])
			elif isinstance(content, list):
				all_stopwords.extend(content)
			else:
				logging.warning("Invalid format in custom stopwords file. Expected a list of stopwords or a dictionary with 'custom_stopwords' key.")
		except Exception as e:
			logging.error(f"Error loading custom stopwords: {str(e)}")
			
	# Deduplicate stopwords
	all_stopwords = list(set(all_stopwords))
	return {"stopwords": all_stopwords}

def load_custom_entities(files: Union[str, UploadedFile, List[UploadedFile]]) -> Dict[str, List[Dict[str, Any]]]:
	"""
	Load custom entities from one or more JSON files.

	Args:
		files (Union[str, UploadedFile, List[UploadedFile]]): The file(s) to load entities from.

	Returns:
		Dict[str, List[Dict[str, Any]]]: A dictionary where keys are entity types and values are lists of standardized entity dictionaries.
	"""
	if isinstance(files, (str, UploadedFile)):
		files = [files]
		
	custom_entities = {}
	for file in files:
		try:
			content = load_json_file(file)
			if isinstance(content, dict):
				for entity_type, entities in content.items():
					if entity_type not in custom_entities:
						custom_entities[entity_type] = []
					if entity_type == 'PERSON':
						custom_entities[entity_type].extend(entities)
					elif isinstance(entities, list):
						for entity in entities:
							if isinstance(entity, dict) and 'term' in entity:
								custom_entities[entity_type].append(entity)
							elif isinstance(entity, str):
								custom_entities[entity_type].append({'term': entity, 'category': [entity_type.lower()]})
							else:
								logging.warning(f"Invalid entity format in {entity_type}: {entity}. Expected a dictionary with 'term' key or a string.")
					else:
						logging.warning(f"Invalid format for entity type {entity_type}. Expected a list of entities.")
			else:
				logging.warning("Invalid format in custom entities file. Expected a dictionary of entity types and entities.")
		except Exception as e:
			logging.error(f"Error loading custom entities from file: {str(e)}")
			
	# Deduplicate entities (except for PERSON category)
	for entity_type in custom_entities:
		if entity_type != 'PERSON':
			custom_entities[entity_type] = list({entity['term']: entity for entity in custom_entities[entity_type]}.values())
			
	return custom_entities

def load_multi_word_entries(files: Union[str, UploadedFile, List[UploadedFile]]) -> Dict[str, List[Dict[str, Any]]]:
	"""
	Load multi-word entries from one or more JSON files.

	Args:
		files (Union[str, UploadedFile, List[UploadedFile]]): The file(s) to load multi-word entries from.

	Returns:
		Dict[str, List[Dict[str, Any]]]: A dictionary where keys are categories and values are lists of multi-word entry dictionaries.
	"""
	if isinstance(files, (str, UploadedFile)):
		files = [files]
		
	multi_word_entries = {}
	for file in files:
		try:
			content = load_json_file(file)
			if isinstance(content, dict):
				for category, entries in content.items():
					if category not in multi_word_entries:
						multi_word_entries[category] = []
					if isinstance(entries, list):
						for entry in entries:
							if isinstance(entry, dict) and 'term' in entry:
								multi_word_entries[category].append(entry)
					else:
						logging.warning(f"Invalid format for category {category}. Expected a list of entry dictionaries.")
			else:
				logging.warning("Invalid format in multi-word entries file. Expected a dictionary of categories and entries.")
		except Exception as e:
			logging.error(f"Error loading multi-word entries from file: {str(e)}")
			
	# Deduplicate entries based on the 'term' field for each category
	for category in multi_word_entries:
		multi_word_entries[category] = list({entry['term']: entry for entry in multi_word_entries[category]}.values())
		
	return multi_word_entries

def load_language_data(files: Union[str, UploadedFile, List[UploadedFile]]) -> Dict[str, List[Dict[str, Any]]]:
	"""
	Load language data from one or more JSON files.

	Args:
		files (Union[str, UploadedFile, List[UploadedFile]]): The file(s) to load language data from.

	Returns:
		Dict[str, List[Dict[str, Any]]]: A dictionary where keys are language data categories and values are lists of
		standardized term dictionaries including pos tags and match types.
	"""
	if isinstance(files, (str, UploadedFile)):
		files = [files]
		
	language_data = {}
	required_fields = {'term', 'category'}
	optional_fields = {'pos', 'match_type'}
	
	for file in files:
		try:
			content = load_json_file(file)
			if isinstance(content, dict):
				for category, terms in content.items():
					if category not in language_data:
						language_data[category] = []
					if isinstance(terms, list):
						for term in terms:
							if isinstance(term, dict):
								# Verify required fields
								if not required_fields.issubset(term.keys()):
									logging.warning(f"Missing required fields in term {term}. Expected {required_fields}")
									continue
								
								# Create standardized term entry
								standardized_term = {
									'term': term['term'],
									'category': term['category'],
									'pos': term.get('pos', ['ADV']),  # Default to ADV if not specified
								}
								
								# Add optional match_type if present
								if 'match_type' in term:
									standardized_term['match_type'] = term['match_type']
								elif ' ' in term['term']:  # Auto-add match_type for multi-word terms
									standardized_term['match_type'] = 'phrase'
								
								language_data[category].append(standardized_term)
								
							elif isinstance(term, str):
								# Create basic term entry with defaults
								language_data[category].append({
									'term': term,
									'category': [category.lower()],
									'pos': ['ADV']  # Default POS tag
								})
							else:
								logging.warning(f"Invalid term format in {category}: {term}. "
												f"Expected a dictionary with required fields or a string.")
					else:
						logging.warning(f"Invalid format for category {category}. Expected a list of terms.")
			else:
				logging.warning(f"Invalid format in {file}. Expected a dictionary of categories and terms.")
		except Exception as e:
			logging.error(f"Error loading language data from {file}: {str(e)}")
			
	# Deduplicate terms while preserving new fields
	for category in language_data:
		language_data[category] = list({term['term']: term for term in language_data[category]}.values())
		
	return language_data

def load_regex_patterns(files: Union[str, UploadedFile, List[Union[str, UploadedFile]]]) -> Dict[str, List[Dict[str, str]]]:
	"""
	Load regex patterns from one or more JSON files.

	Args:
		files (Union[str, UploadedFile, List[Union[str, UploadedFile]]]): The file(s) to load patterns from.

	Returns:
		Dict[str, List[Dict[str, str]]]: A dictionary where keys are pattern categories and values are lists of pattern dictionaries.
	"""
	if isinstance(files, (str, UploadedFile)):
		files = [files]
		
	patterns = {}
	for file in files:
		try:
			content = load_json_file(file)
			if isinstance(content, dict):
				for category, category_patterns in content.items():
					if category not in patterns:
						patterns[category] = []
					if isinstance(category_patterns, list):
						patterns[category].extend(category_patterns)
					else:
						logging.warning(f"Invalid format for category {category}. Expected a list of pattern dictionaries.")
			else:
				logging.warning("Invalid format in regex patterns file. Expected a dictionary of pattern categories.")
		except Exception as e:
			logging.error(f"Error loading regex patterns: {str(e)}")
			
	return patterns

def load_aspect_configuration(files: Union[str, UploadedFile, List[UploadedFile]]) -> Dict[str, Dict[str, List[str]]]:
	"""
	Load aspect configuration from one or more JSON files.

	Args:
		files (Union[str, UploadedFile, List[UploadedFile]]): The file(s) to load configurations from.

	Returns:
		Dict[str, Dict[str, List[str]]]: Combined aspect configuration dictionary
	"""
	if isinstance(files, (str, UploadedFile)):
		files = [files]
		
	aspect_config = {}
	for file in files:
		try:
			content = load_json_file(file)
			if isinstance(content, dict):
				for category, subcategories in content.items():
					if category not in aspect_config:
						aspect_config[category] = {}
					for subcat, terms in subcategories.items():
						if subcat not in aspect_config[category]:
							aspect_config[category][subcat] = []
						aspect_config[category][subcat].extend(terms)
			else:
				logging.warning(f"Invalid format in aspect configuration file. Expected a dictionary.")
		except Exception as e:
			logging.error(f"Error loading aspect configuration: {str(e)}")
			
	# Deduplicate terms in each subcategory
	for category in aspect_config:
		for subcat in aspect_config[category]:
			aspect_config[category][subcat] = list(set(aspect_config[category][subcat]))
			
	return aspect_config