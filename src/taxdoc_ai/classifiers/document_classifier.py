"""
Tax Document Classifier using LegalBERT for document type classification.

This module implements a neural network classifier that can identify different types
of tax documents (W-2, 1099-R, 1099-MISC, etc.) using the LegalBERT model.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Document type mappings
DOCUMENT_TYPES = [
    'W-2',
    '1099-R', 
    '1099-MISC',
    '1099-DIV',
    'Receipt',
    'Bank Statement',
    '1040',
    'Schedule A'
]

class TaxDocumentClassifier(nn.Module):
    """
    Neural network classifier for tax document types using LegalBERT.
    
    This classifier takes text extracted from tax documents and classifies them
    into one of the predefined document types using a fine-tuned LegalBERT model.
    """
    
    def __init__(self, num_classes: int = 8, model_name: str = "jhu-clsp/LegalBert", dropout_rate: float = 0.3):
        """
        Initialize the TaxDocumentClassifier.
        
        Args:
            num_classes: Number of document types to classify (default: 8)
            model_name: Name of the pre-trained LegalBERT model
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        
        # Load pre-trained LegalBERT model
        try:
            self.legalbert = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to load LegalBERT model: {e}")
            raise
        
        # Get the hidden size from the model
        self.hidden_size = self.legalbert.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            input_ids: Tokenized input text
            attention_mask: Attention mask for the input
            
        Returns:
            Classification logits
        """
        # Get LegalBERT outputs
        outputs = self.legalbert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use pooled output for classification
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classification
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        
        return logits
    
    def predict(self, text: str, return_probabilities: bool = False) -> Union[str, Dict[str, float]]:
        """
        Predict document type for given text.
        
        Args:
            text: Input text to classify
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predicted document type or dictionary with probabilities
        """
        self.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True, 
            padding=True
        )
        
        with torch.no_grad():
            logits = self.forward(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.softmax(logits, dim=-1)
            
            if return_probabilities:
                # Return probabilities for all classes
                prob_dict = {}
                for i, doc_type in enumerate(DOCUMENT_TYPES):
                    prob_dict[doc_type] = probabilities[0][i].item()
                return prob_dict
            else:
                # Return predicted class
                predicted_class = torch.argmax(logits, dim=-1).item()
                return DOCUMENT_TYPES[predicted_class]
    
    def predict_batch(self, texts: List[str], return_probabilities: bool = False) -> Union[List[str], List[Dict[str, float]]]:
        """
        Predict document types for a batch of texts.
        
        Args:
            texts: List of input texts to classify
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of predicted document types or list of probability dictionaries
        """
        self.eval()
        
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            logits = self.forward(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.softmax(logits, dim=-1)
            
            if return_probabilities:
                # Return probabilities for all classes
                results = []
                for i in range(len(texts)):
                    prob_dict = {}
                    for j, doc_type in enumerate(DOCUMENT_TYPES):
                        prob_dict[doc_type] = probabilities[i][j].item()
                    results.append(prob_dict)
                return results
            else:
                # Return predicted classes
                predicted_classes = torch.argmax(logits, dim=-1)
                return [DOCUMENT_TYPES[idx.item()] for idx in predicted_classes]
    
    def get_confidence_score(self, text: str) -> float:
        """
        Get confidence score for the prediction.
        
        Args:
            text: Input text to classify
            
        Returns:
            Confidence score (0-1)
        """
        probabilities = self.predict(text, return_probabilities=True)
        max_prob = max(probabilities.values())
        return max_prob
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'model_name': self.model_name,
            'dropout_rate': self.dropout_rate,
            'document_types': DOCUMENT_TYPES
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'TaxDocumentClassifier':
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded TaxDocumentClassifier instance
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Create model instance
        model = cls(
            num_classes=checkpoint['num_classes'],
            model_name=checkpoint['model_name'],
            dropout_rate=checkpoint['dropout_rate']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def get_model_info(self) -> Dict[str, Union[str, int, float]]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'hidden_size': self.hidden_size,
            'dropout_rate': self.dropout_rate,
            'document_types': DOCUMENT_TYPES,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
