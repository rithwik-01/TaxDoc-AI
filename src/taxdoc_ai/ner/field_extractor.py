"""
Tax Field Extractor using Named Entity Recognition.

This module implements NER-based extraction of tax-specific fields from document text
using fine-tuned LegalBERT models for tax document processing.
"""

import re
import logging
from typing import Dict, List, Optional, Union, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import numpy as np

logger = logging.getLogger(__name__)

# Tax-specific entity types
TAX_ENTITY_TYPES = [
    'SSN',
    'EMPLOYER_NAME', 
    'WAGE_AMOUNT',
    'TAX_WITHHELD',
    'ADDRESS',
    'DATE',
    'EMPLOYER_EIN',
    'EMPLOYEE_NAME',
    'FEDERAL_TAX',
    'STATE_TAX',
    'SOCIAL_SECURITY_TAX',
    'MEDICARE_TAX',
    'ACCOUNT_NUMBER',
    'ROUTING_NUMBER',
    'PHONE_NUMBER',
    'EMAIL'
]

class TaxFieldExtractor:
    """
    Named Entity Recognition extractor for tax-specific fields.
    
    This class uses fine-tuned LegalBERT models to extract structured information
    from tax document text, identifying key fields like SSN, amounts, dates, etc.
    """
    
    def __init__(self, model_path: Optional[str] = None, model_name: str = "jhu-clsp/LegalBert"):
        """
        Initialize the TaxFieldExtractor.
        
        Args:
            model_path: Path to fine-tuned NER model (if available)
            model_name: Base model name for tokenizer
        """
        self.model_name = model_name
        self.model_path = model_path
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model (fine-tuned if available, otherwise base model)
            if model_path and model_path.exists():
                self.model = AutoModelForTokenClassification.from_pretrained(model_path)
                logger.info(f"Loaded fine-tuned NER model from {model_path}")
            else:
                # Use base model with custom labels
                self.model = AutoModelForTokenClassification.from_pretrained(
                    model_name,
                    num_labels=len(TAX_ENTITY_TYPES) * 2 + 1  # BIO tags + O
                )
                logger.warning("Using base model - consider fine-tuning for better performance")
            
            # Create NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize TaxFieldExtractor: {e}")
            raise
    
    def extract_tax_entities(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract tax-specific entities from text.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary mapping entity types to lists of extracted entities
        """
        try:
            # Run NER pipeline
            entities = self.ner_pipeline(text)
            
            # Initialize result structure
            tax_fields = {entity_type: [] for entity_type in TAX_ENTITY_TYPES}
            
            # Process entities
            for entity in entities:
                entity_type = entity.get('entity_group', '').upper()
                
                # Map to our tax entity types
                mapped_type = self._map_entity_type(entity_type)
                
                if mapped_type in tax_fields:
                    tax_fields[mapped_type].append({
                        'text': entity['word'],
                        'confidence': entity['score'],
                        'start': entity['start'],
                        'end': entity['end']
                    })
            
            # Apply post-processing rules
            tax_fields = self._apply_post_processing_rules(text, tax_fields)
            
            logger.info(f"Extracted entities: {sum(len(entities) for entities in tax_fields.values())} total")
            return tax_fields
            
        except Exception as e:
            logger.error(f"Error extracting tax entities: {e}")
            return {entity_type: [] for entity_type in TAX_ENTITY_TYPES}
    
    def _map_entity_type(self, entity_type: str) -> str:
        """
        Map generic entity types to tax-specific types.
        
        Args:
            entity_type: Generic entity type from model
            
        Returns:
            Mapped tax-specific entity type
        """
        mapping = {
            'PERSON': 'EMPLOYEE_NAME',
            'ORG': 'EMPLOYER_NAME',
            'MONEY': 'WAGE_AMOUNT',
            'DATE': 'DATE',
            'LOC': 'ADDRESS',
            'PHONE': 'PHONE_NUMBER',
            'EMAIL': 'EMAIL'
        }
        
        return mapping.get(entity_type, entity_type)
    
    def _apply_post_processing_rules(self, text: str, tax_fields: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Apply post-processing rules to improve entity extraction.
        
        Args:
            text: Original text
            tax_fields: Extracted entities
            
        Returns:
            Post-processed entities
        """
        # Apply regex-based extraction for specific patterns
        tax_fields = self._extract_ssn_patterns(text, tax_fields)
        tax_fields = self._extract_currency_patterns(text, tax_fields)
        tax_fields = self._extract_date_patterns(text, tax_fields)
        tax_fields = self._extract_ein_patterns(text, tax_fields)
        
        # Remove duplicates and low-confidence entities
        tax_fields = self._deduplicate_entities(tax_fields)
        
        return tax_fields
    
    def _extract_ssn_patterns(self, text: str, tax_fields: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Extract SSN patterns using regex."""
        ssn_patterns = [
            r'\b\d{3}-?\d{2}-?\d{4}\b',  # Standard SSN format
            r'\b\d{9}\b'  # 9 consecutive digits
        ]
        
        for pattern in ssn_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                ssn_text = match.group()
                # Avoid adding duplicates
                if not any(entity['text'] == ssn_text for entity in tax_fields['SSN']):
                    tax_fields['SSN'].append({
                        'text': ssn_text,
                        'confidence': 0.9,
                        'start': match.start(),
                        'end': match.end()
                    })
        
        return tax_fields
    
    def _extract_currency_patterns(self, text: str, tax_fields: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Extract currency amounts using regex."""
        currency_patterns = [
            r'\$[\d,]+\.?\d*',  # $1,234.56
            r'\b\d{1,3}(?:,\d{3})*\.?\d*\b'  # 1,234.56
        ]
        
        for pattern in currency_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                amount_text = match.group()
                # Avoid adding duplicates
                if not any(entity['text'] == amount_text for entity in tax_fields['WAGE_AMOUNT']):
                    tax_fields['WAGE_AMOUNT'].append({
                        'text': amount_text,
                        'confidence': 0.8,
                        'start': match.start(),
                        'end': match.end()
                    })
        
        return tax_fields
    
    def _extract_date_patterns(self, text: str, tax_fields: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Extract date patterns using regex."""
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',  # MM-DD-YYYY
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',  # YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_text = match.group()
                # Avoid adding duplicates
                if not any(entity['text'] == date_text for entity in tax_fields['DATE']):
                    tax_fields['DATE'].append({
                        'text': date_text,
                        'confidence': 0.8,
                        'start': match.start(),
                        'end': match.end()
                    })
        
        return tax_fields
    
    def _extract_ein_patterns(self, text: str, tax_fields: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Extract EIN patterns using regex."""
        ein_patterns = [
            r'\b\d{2}-?\d{7}\b',  # Standard EIN format
            r'\b\d{9}\b'  # 9 consecutive digits (context-dependent)
        ]
        
        for pattern in ein_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                ein_text = match.group()
                # Check if it's likely an EIN (not SSN)
                if not self._is_likely_ssn(ein_text):
                    # Avoid adding duplicates
                    if not any(entity['text'] == ein_text for entity in tax_fields['EMPLOYER_EIN']):
                        tax_fields['EMPLOYER_EIN'].append({
                            'text': ein_text,
                            'confidence': 0.7,
                            'start': match.start(),
                            'end': match.end()
                        })
        
        return tax_fields
    
    def _is_likely_ssn(self, number: str) -> bool:
        """Check if a 9-digit number is likely an SSN based on patterns."""
        # Remove non-digits
        digits = re.sub(r'\D', '', number)
        
        if len(digits) != 9:
            return False
        
        # SSN patterns to avoid (invalid SSN ranges)
        first_three = int(digits[:3])
        if first_three in [0, 666] or first_three >= 900:
            return False
        
        # Check for common SSN patterns
        if digits[:3] == '000' or digits[3:5] == '00' or digits[5:] == '0000':
            return False
        
        return True
    
    def _deduplicate_entities(self, tax_fields: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Remove duplicate entities and filter by confidence."""
        for entity_type, entities in tax_fields.items():
            # Remove duplicates based on text content
            seen_texts = set()
            unique_entities = []
            
            for entity in entities:
                if entity['text'] not in seen_texts and entity['confidence'] > 0.3:
                    seen_texts.add(entity['text'])
                    unique_entities.append(entity)
            
            # Sort by confidence (highest first)
            tax_fields[entity_type] = sorted(unique_entities, key=lambda x: x['confidence'], reverse=True)
        
        return tax_fields
    
    def extract_structured_data(self, text: str) -> Dict[str, Union[str, float, None]]:
        """
        Extract structured data in a simplified format.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary with extracted fields and their best values
        """
        entities = self.extract_tax_entities(text)
        
        structured_data = {}
        for field_name, field_entities in entities.items():
            if field_entities:
                # Take the highest confidence entity
                best_entity = max(field_entities, key=lambda x: x['confidence'])
                structured_data[field_name] = {
                    'value': best_entity['text'],
                    'confidence': best_entity['confidence']
                }
            else:
                structured_data[field_name] = {
                    'value': None,
                    'confidence': 0.0
                }
        
        return structured_data
    
    def get_extraction_summary(self, text: str) -> Dict[str, Union[int, float]]:
        """
        Get summary statistics for entity extraction.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary with extraction statistics
        """
        entities = self.extract_tax_entities(text)
        
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        high_confidence_entities = sum(
            len([e for e in entity_list if e['confidence'] > 0.7])
            for entity_list in entities.values()
        )
        
        return {
            'total_entities': total_entities,
            'high_confidence_entities': high_confidence_entities,
            'confidence_ratio': high_confidence_entities / max(total_entities, 1),
            'fields_found': len([field for field, entities in entities.items() if entities])
        }
