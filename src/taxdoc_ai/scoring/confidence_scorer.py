"""
Confidence scoring system for tax document extraction results.

This module implements a comprehensive confidence scoring system that combines
OCR confidence, NER model confidence, and business logic validation to provide
reliable confidence scores for extracted tax fields.
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Callable
import numpy as np
from datetime import datetime, date
import calendar

logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    """Data class representing an extraction result with confidence scores."""
    
    field_name: str
    value: str
    confidence: float
    source_confidence: float  # OCR confidence
    model_confidence: float   # NER confidence
    validation_score: float   # Business logic validation
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    metadata: Optional[Dict] = None

class ConfidenceScorer:
    """
    Confidence scoring system for tax field extraction.
    
    This class combines multiple confidence sources to provide reliable
    confidence scores for extracted tax document fields.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the confidence scorer.
        
        Args:
            weights: Weights for different confidence components
        """
        self.weights = weights or {
            'ocr': 0.3,
            'ner': 0.4,
            'validation': 0.3
        }
        
        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Confidence weights sum to {total_weight}, normalizing to 1.0")
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        # Initialize validators
        self.validators = self._initialize_validators()
    
    def _initialize_validators(self) -> Dict[str, Callable[[str], float]]:
        """Initialize field-specific validators."""
        return {
            'SSN': self._validate_ssn,
            'EMPLOYER_EIN': self._validate_ein,
            'WAGE_AMOUNT': self._validate_currency,
            'TAX_WITHHELD': self._validate_currency,
            'FEDERAL_TAX': self._validate_currency,
            'STATE_TAX': self._validate_currency,
            'SOCIAL_SECURITY_TAX': self._validate_currency,
            'MEDICARE_TAX': self._validate_currency,
            'DATE': self._validate_date,
            'PHONE_NUMBER': self._validate_phone,
            'EMAIL': self._validate_email,
            'ACCOUNT_NUMBER': self._validate_account_number,
            'ROUTING_NUMBER': self._validate_routing_number,
            'EMPLOYER_NAME': self._validate_name,
            'EMPLOYEE_NAME': self._validate_name,
            'ADDRESS': self._validate_address
        }
    
    def calculate_composite_confidence(self, result: ExtractionResult) -> float:
        """
        Calculate weighted composite confidence score.
        
        Args:
            result: Extraction result with individual confidence scores
            
        Returns:
            Composite confidence score (0-1)
        """
        composite_score = (
            result.source_confidence * self.weights['ocr'] +
            result.model_confidence * self.weights['ner'] +
            result.validation_score * self.weights['validation']
        )
        
        return min(composite_score, 1.0)
    
    def validate_tax_field(self, field_name: str, value: str) -> float:
        """
        Validate tax field using business logic.
        
        Args:
            field_name: Name of the field to validate
            value: Value to validate
            
        Returns:
            Validation confidence score (0-1)
        """
        if not value or not value.strip():
            return 0.0
        
        validator = self.validators.get(field_name)
        if validator:
            try:
                return validator(value.strip())
            except Exception as e:
                logger.warning(f"Validation error for {field_name}: {e}")
                return 0.5
        else:
            # Default moderate confidence for unknown fields
            return 0.5
    
    def _validate_ssn(self, ssn: str) -> float:
        """Validate SSN format and ranges."""
        # Remove all non-digits
        digits = re.sub(r'\D', '', ssn)
        
        if len(digits) != 9:
            return 0.0
        
        # Check for invalid SSN patterns
        first_three = int(digits[:3])
        middle_two = int(digits[3:5])
        last_four = int(digits[5:])
        
        # Invalid ranges
        if first_three == 0 or first_three == 666 or first_three >= 900:
            return 0.0
        
        if middle_two == 0 or last_four == 0:
            return 0.0
        
        # Check for sequential patterns (likely fake)
        if self._is_sequential(digits):
            return 0.2
        
        # Check for repeated patterns
        if self._has_repeated_patterns(digits):
            return 0.3
        
        return 0.9
    
    def _validate_ein(self, ein: str) -> float:
        """Validate EIN format."""
        # Remove all non-digits
        digits = re.sub(r'\D', '', ein)
        
        if len(digits) != 9:
            return 0.0
        
        # EIN format: XX-XXXXXXX
        first_two = int(digits[:2])
        
        # Valid EIN prefixes (simplified check)
        valid_prefixes = list(range(10, 99))  # Most EINs start with 10-99
        if first_two not in valid_prefixes:
            return 0.3
        
        return 0.8
    
    def _validate_currency(self, amount: str) -> float:
        """Validate currency amount format and range."""
        try:
            # Remove currency symbols and commas
            clean_amount = re.sub(r'[$,]', '', amount)
            
            # Handle negative amounts
            is_negative = clean_amount.startswith('-')
            if is_negative:
                clean_amount = clean_amount[1:]
            
            # Parse as float
            float_amount = float(clean_amount)
            
            # Check reasonable ranges for tax amounts
            if float_amount < 0:
                return 0.3  # Negative amounts are possible but less common
            
            if float_amount > 10000000:  # $10M seems like an upper bound
                return 0.2
            
            # Check for too many decimal places
            if '.' in clean_amount and len(clean_amount.split('.')[1]) > 2:
                return 0.4
            
            return 0.9
            
        except (ValueError, TypeError):
            return 0.1
    
    def _validate_date(self, date_str: str) -> float:
        """Validate date format and reasonableness."""
        try:
            # Common date formats
            date_formats = [
                '%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d',
                '%m/%d/%y', '%m-%d-%y', '%y-%m-%d',
                '%B %d, %Y', '%b %d, %Y',
                '%d %B %Y', '%d %b %Y'
            ]
            
            parsed_date = None
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt).date()
                    break
                except ValueError:
                    continue
            
            if parsed_date is None:
                return 0.0
            
            # Check reasonable date ranges (1900 to current year + 1)
            current_year = date.today().year
            if parsed_date.year < 1900 or parsed_date.year > current_year + 1:
                return 0.2
            
            # Check for future dates (might be reasonable for some tax documents)
            if parsed_date > date.today():
                return 0.6
            
            return 0.9
            
        except Exception:
            return 0.1
    
    def _validate_phone(self, phone: str) -> float:
        """Validate phone number format."""
        # Remove all non-digits
        digits = re.sub(r'\D', '', phone)
        
        # US phone numbers should be 10 digits
        if len(digits) == 10:
            return 0.9
        elif len(digits) == 11 and digits[0] == '1':
            return 0.8
        else:
            return 0.2
    
    def _validate_email(self, email: str) -> float:
        """Validate email format."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, email):
            return 0.9
        else:
            return 0.1
    
    def _validate_account_number(self, account: str) -> float:
        """Validate bank account number format."""
        # Remove all non-digits
        digits = re.sub(r'\D', '', account)
        
        # Account numbers are typically 8-17 digits
        if 8 <= len(digits) <= 17:
            return 0.8
        else:
            return 0.3
    
    def _validate_routing_number(self, routing: str) -> float:
        """Validate bank routing number format."""
        # Remove all non-digits
        digits = re.sub(r'\D', '', routing)
        
        # Routing numbers are exactly 9 digits
        if len(digits) == 9:
            # Basic checksum validation (simplified)
            if self._validate_routing_checksum(digits):
                return 0.9
            else:
                return 0.6
        else:
            return 0.1
    
    def _validate_routing_checksum(self, routing: str) -> bool:
        """Validate routing number checksum."""
        try:
            # Routing number checksum algorithm
            digits = [int(d) for d in routing]
            checksum = (
                3 * (digits[0] + digits[3] + digits[6]) +
                7 * (digits[1] + digits[4] + digits[7]) +
                1 * (digits[2] + digits[5] + digits[8])
            ) % 10
            
            return checksum == 0
        except:
            return False
    
    def _validate_name(self, name: str) -> float:
        """Validate name format."""
        # Names should contain letters and common name characters
        if len(name) < 2:
            return 0.1
        
        # Check for reasonable name patterns
        if re.match(r'^[A-Za-z\s\.,\'-]+$', name):
            # Check for too many special characters
            special_chars = len(re.findall(r'[^\w\s]', name))
            if special_chars > len(name) * 0.3:
                return 0.3
            
            return 0.8
        else:
            return 0.2
    
    def _validate_address(self, address: str) -> float:
        """Validate address format."""
        if len(address) < 10:
            return 0.2
        
        # Check for common address patterns
        if re.search(r'\d+', address):  # Should contain numbers
            return 0.7
        else:
            return 0.3
    
    def _is_sequential(self, digits: str) -> bool:
        """Check if digits form a sequential pattern."""
        for i in range(len(digits) - 2):
            if (int(digits[i+1]) == int(digits[i]) + 1 and 
                int(digits[i+2]) == int(digits[i]) + 2):
                return True
        return False
    
    def _has_repeated_patterns(self, digits: str) -> bool:
        """Check for repeated digit patterns."""
        # Check for repeated sequences
        for length in range(2, len(digits) // 2 + 1):
            for i in range(len(digits) - length * 2 + 1):
                pattern = digits[i:i+length]
                if digits[i+length:i+length*2] == pattern:
                    return True
        return False
    
    def score_extraction_batch(self, results: List[ExtractionResult]) -> List[ExtractionResult]:
        """
        Score a batch of extraction results.
        
        Args:
            results: List of extraction results to score
            
        Returns:
            List of results with updated confidence scores
        """
        scored_results = []
        
        for result in results:
            # Calculate validation score
            result.validation_score = self.validate_tax_field(result.field_name, result.value)
            
            # Calculate composite confidence
            result.confidence = self.calculate_composite_confidence(result)
            
            scored_results.append(result)
        
        return scored_results
    
    def get_confidence_summary(self, results: List[ExtractionResult]) -> Dict[str, Union[float, int]]:
        """
        Get summary statistics for confidence scores.
        
        Args:
            results: List of extraction results
            
        Returns:
            Dictionary with confidence statistics
        """
        if not results:
            return {
                'total_results': 0,
                'average_confidence': 0.0,
                'high_confidence_count': 0,
                'medium_confidence_count': 0,
                'low_confidence_count': 0
            }
        
        confidences = [r.confidence for r in results]
        
        high_confidence = sum(1 for c in confidences if c >= 0.8)
        medium_confidence = sum(1 for c in confidences if 0.5 <= c < 0.8)
        low_confidence = sum(1 for c in confidences if c < 0.5)
        
        return {
            'total_results': len(results),
            'average_confidence': np.mean(confidences),
            'high_confidence_count': high_confidence,
            'medium_confidence_count': medium_confidence,
            'low_confidence_count': low_confidence,
            'confidence_std': np.std(confidences)
        }
