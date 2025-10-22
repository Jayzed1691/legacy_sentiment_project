# FinancialEntityRefiner.py

import re
from typing import List, Tuple, Dict, Any

class FinancialEntityRefiner:
    def __init__(self, language_data: Dict[str, List[Dict[str, Any]]]):
        self.currency_pattern = re.compile(r'(\b(?:USD|Rp\.?|IDR|EUR|GBP|JPY)\s?)?(\d+(?:,\d{3})*(?:\.\d+)?)\s?(million|billion|trillion|rupiah)?', re.IGNORECASE)
        self.percentage_pattern = re.compile(r'(\d+(?:\.\d+)?)\s?(percent|%)', re.IGNORECASE)
        self.date_pattern = re.compile(r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b')
        self.time_comparison_pattern = re.compile(r'\b(year-on-year|month-on-month|quarter-on-quarter)\b', re.IGNORECASE)
        self.quantitative_modifiers = set(item['term'].lower() for item in language_data.get('quantitative_modifiers', []))
        
    def refine_entities(self, text: str, entities: List[Tuple[str, str, str, int, int]]) -> List[Tuple[str, str, str, int, int]]:
        refined_entities = []
        for entity in entities:
            entity_text, entity_type, source, start, end = entity
            
            # Handle percentage pairs
            if ' and ' in entity_text:
                parts = entity_text.split(' and ')
                if len(parts) == 2 and all(self.percentage_pattern.search(part) for part in parts):
                    for part in parts:
                        part = part.strip()
                        part_start = start + entity_text.index(part)
                        part_end = part_start + len(part)
                        refined_entities.append((part, 'PERCENT', 'Refined', part_start, part_end))
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