# named_entity_recognition.py - proposed by Gemini Advnaced

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.util import filter_spans
from typing import List, Tuple, Dict, Any, Union
from streamlit.runtime.uploaded_file_manager import UploadedFile
from file_utils import load_custom_entities

@Language.factory("custom_entity_matcher")
class CustomEntityMatcher:
    def __init__(self, nlp: Language, name: str, custom_entities: Dict[str, List[Any]]):
        self.name = name
        self.nlp = nlp
        self.custom_entities = custom_entities
        self.matcher = spacy.matcher.PhraseMatcher(nlp.vocab)
        for entity_type, phrases in custom_entities.items():
            patterns = []
            for phrase in phrases:
                if isinstance(phrase, dict):
                    if 'term' in phrase:
                        patterns.append(nlp.make_doc(phrase['term']))
                    elif 'full_name' in phrase:  # Check for full_name and variations
                        patterns.append(nlp.make_doc(phrase['full_name']))
                        patterns.extend([nlp.make_doc(var) for var in phrase.get('variations', [])])
                elif isinstance(phrase, str):
                    patterns.append(nlp.make_doc(phrase))
            self.matcher.add(entity_type, patterns)
            
    def __call__(self, doc):
        matches = self.matcher(doc)
        new_ents = []
        for match_id, start, end in matches:
            entity_type = doc.vocab.strings[match_id]
            span = Span(doc, start, end, label=entity_type)
            new_ents.append(span)
            
        # Combine existing and new entities
        ents = list(doc.ents) + new_ents
        
        # Filter and sort spans
        filtered_ents = filter_spans(ents)
        
        # Set the new entities
        doc.ents = filtered_ents
        return doc
    
class NamedEntityRecognizer:
    def __init__(self, model: str = "en_core_web_sm", custom_entities_files: Union[str, UploadedFile, List[UploadedFile]] = None):
        self.nlp = spacy.load(model)
        self.custom_entities = load_custom_entities(custom_entities_files) if custom_entities_files else {}
        
        # Add the custom entity matcher to the pipeline
        self.nlp.add_pipe("custom_entity_matcher", config={"custom_entities": self.custom_entities}, last=True)
        
    def process_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Process the input text and return a list of named entities.

        Args:
            text (str): The input text to process.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing entity text and entity label.
        """
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    
    def get_entity_counts(self, entities: List[Tuple[str, str]]) -> Dict[str, int]:
        """
        Count the occurrences of each entity type.

        Args:
            entities (List[Tuple[str, str]]): A list of entities and their labels.

        Returns:
            Dict[str, int]: A dictionary with entity types as keys and their counts as values.
        """
        return {ent_type: sum(1 for _, label in entities if label == ent_type) for ent_type in set(label for _, label in entities)}
    
    def get_most_frequent_entities(self, entities: List[Tuple[str, str]], top_n: int = 10) -> List[Tuple[str, str, int]]:
        """
        Get the most frequently occurring entities.

        Args:
            entities (List[Tuple[str, str]]): A list of entities and their labels.
            top_n (int): The number of top entities to return.

        Returns:
            List[Tuple[str, str, int]]: A list of tuples containing entity text, label, and count.
        """
        from collections import Counter
        entity_counter = Counter(entities)
        return entity_counter.most_common(top_n)
    
def run_ner_analysis(text: str, custom_entities_files: List[UploadedFile] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run Named Entity Recognition analysis on the given text.

    Args:
        text (str): The input text to analyze.
        custom_entities_file (Optional[str]): Path to a JSON file containing custom entities.

    Returns:
        Dict[str, Any]: A dictionary containing the NER analysis results.
    """
    # Run overall NER analysis
    results = run_ner_analysis_core(text, custom_entities_files)
    
    # Add speaker-level NER if available
    if context and 'transcript_structure' in context and 'transcript_data' in context:
        speakers = context['transcript_structure']['speakers']
        speaker_ner = {}
        for speaker in speakers:
            speaker_text = " ".join(context['transcript_data'].speakers[speaker])
            speaker_ner[speaker] = run_ner_analysis_core(speaker_text, custom_entities_files)
        results['speaker_ner'] = speaker_ner
        
    # Add section-level NER if available
    if context and 'transcript_structure' in context and 'transcript_data' in context:
        sections = context['transcript_structure']['sections']
        section_ner = {}
        for section in sections:
            section_text = " ".join([dialogue.text for s in context['transcript_data'].sections if s.name == section for dialogue in s.dialogues])
            section_ner[section] = run_ner_analysis_core(section_text, custom_entities_files)
        results['section_ner'] = section_ner
        
    return results