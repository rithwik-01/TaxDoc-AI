"""
Evaluation metrics for tax document processing system.

This module implements comprehensive evaluation metrics for document classification,
named entity recognition, and end-to-end pipeline performance.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score
)
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

class ClassificationMetrics:
    """Metrics for document classification evaluation."""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.true_labels = []
        self.confidences = []
        self.processing_times = []
    
    def update(self, prediction: str, true_label: str, confidence: float, processing_time: float):
        """Update metrics with new prediction."""
        self.predictions.append(prediction)
        self.true_labels.append(true_label)
        self.confidences.append(confidence)
        self.processing_times.append(processing_time)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all classification metrics."""
        if not self.predictions:
            return {}
        
        # Convert to indices
        pred_indices = [self.class_names.index(p) for p in self.predictions]
        true_indices = [self.class_names.index(t) for t in self.true_labels]
        
        # Basic metrics
        accuracy = accuracy_score(true_indices, pred_indices)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_indices, pred_indices, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            true_indices, pred_indices, average=None, zero_division=0
        )
        
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1': f1_per_class[i],
                'support': support_per_class[i]
            }
        
        # Confidence metrics
        avg_confidence = np.mean(self.confidences)
        confidence_std = np.std(self.confidences)
        
        # Processing time metrics
        avg_processing_time = np.mean(self.processing_times)
        processing_time_std = np.std(self.processing_times)
        
        # Confusion matrix
        cm = confusion_matrix(true_indices, pred_indices)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'confidence_metrics': {
                'average_confidence': avg_confidence,
                'confidence_std': confidence_std,
                'min_confidence': np.min(self.confidences),
                'max_confidence': np.max(self.confidences)
            },
            'processing_time_metrics': {
                'average_time': avg_processing_time,
                'time_std': processing_time_std,
                'min_time': np.min(self.processing_times),
                'max_time': np.max(self.processing_times)
            },
            'total_samples': len(self.predictions)
        }

class NERMetrics:
    """Metrics for named entity recognition evaluation."""
    
    def __init__(self, entity_types: List[str]):
        self.entity_types = entity_types
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []  # List of predicted entities per document
        self.true_labels = []  # List of true entities per document
        self.confidences = []
    
    def update(self, predicted_entities: Dict[str, List[Dict]], true_entities: Dict[str, List[Dict]], confidence: float):
        """Update metrics with new NER results."""
        self.predictions.append(predicted_entities)
        self.true_labels.append(true_entities)
        self.confidences.append(confidence)
    
    def _extract_entities(self, entities_dict: Dict[str, List[Dict]]) -> List[Tuple[str, str, int, int]]:
        """Extract entities in (type, text, start, end) format."""
        extracted = []
        for entity_type, entity_list in entities_dict.items():
            for entity in entity_list:
                extracted.append((
                    entity_type,
                    entity['text'],
                    entity.get('start', 0),
                    entity.get('end', len(entity['text']))
                ))
        return extracted
    
    def _compute_entity_metrics(self, pred_entities: List[Tuple], true_entities: List[Tuple]) -> Dict[str, int]:
        """Compute entity-level metrics."""
        # Convert to sets for comparison
        pred_set = set(pred_entities)
        true_set = set(true_entities)
        
        # Compute metrics
        true_positives = len(pred_set & true_set)
        false_positives = len(pred_set - true_set)
        false_negatives = len(true_set - pred_set)
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all NER metrics."""
        if not self.predictions:
            return {}
        
        # Overall metrics
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        # Per-entity-type metrics
        entity_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        # Exact match metrics
        exact_matches = 0
        partial_matches = 0
        
        for pred_entities_dict, true_entities_dict in zip(self.predictions, self.true_labels):
            pred_entities = self._extract_entities(pred_entities_dict)
            true_entities = self._extract_entities(true_entities_dict)
            
            # Overall metrics
            metrics = self._compute_entity_metrics(pred_entities, true_entities)
            total_tp += metrics['true_positives']
            total_fp += metrics['false_positives']
            total_fn += metrics['false_negatives']
            
            # Per-entity-type metrics
            for entity_type in self.entity_types:
                pred_type_entities = [e for e in pred_entities if e[0] == entity_type]
                true_type_entities = [e for e in true_entities if e[0] == entity_type]
                
                type_metrics = self._compute_entity_metrics(pred_type_entities, true_type_entities)
                entity_metrics[entity_type]['tp'] += type_metrics['true_positives']
                entity_metrics[entity_type]['fp'] += type_metrics['false_positives']
                entity_metrics[entity_type]['fn'] += type_metrics['false_negatives']
            
            # Exact match (all entities correct)
            if pred_entities == true_entities:
                exact_matches += 1
            
            # Partial match (some entities correct)
            if metrics['true_positives'] > 0:
                partial_matches += 1
        
        # Compute overall precision, recall, F1
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Compute per-entity-type metrics
        per_entity_metrics = {}
        for entity_type, metrics in entity_metrics.items():
            tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_entity_metrics[entity_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            }
        
        # Confidence metrics
        avg_confidence = np.mean(self.confidences) if self.confidences else 0
        
        return {
            'overall_metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': total_tp,
                'false_positives': total_fp,
                'false_negatives': total_fn
            },
            'per_entity_metrics': per_entity_metrics,
            'match_metrics': {
                'exact_match_accuracy': exact_matches / len(self.predictions),
                'partial_match_accuracy': partial_matches / len(self.predictions)
            },
            'confidence_metrics': {
                'average_confidence': avg_confidence,
                'confidence_std': np.std(self.confidences) if self.confidences else 0
            },
            'total_documents': len(self.predictions)
        }

class PipelineMetrics:
    """Metrics for end-to-end pipeline evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.results = []
        self.processing_times = []
        self.user_corrections = []
    
    def update(self, result: Dict[str, Any], processing_time: float, user_corrections: Optional[int] = None):
        """Update metrics with new pipeline result."""
        self.results.append(result)
        self.processing_times.append(processing_time)
        self.user_corrections.append(user_corrections or 0)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all pipeline metrics."""
        if not self.results:
            return {}
        
        # Processing time metrics
        avg_processing_time = np.mean(self.processing_times)
        processing_time_std = np.std(self.processing_times)
        
        # Confidence metrics
        all_confidences = []
        for result in self.results:
            if 'confidence_summary' in result:
                all_confidences.append(result['confidence_summary'].get('average_confidence', 0))
        
        avg_confidence = np.mean(all_confidences) if all_confidences else 0
        
        # Field extraction metrics
        total_fields = 0
        high_confidence_fields = 0
        
        for result in self.results:
            if 'extracted_fields' in result:
                fields = result['extracted_fields']
                total_fields += len(fields)
                high_confidence_fields += sum(1 for f in fields if f.get('confidence', 0) > 0.8)
        
        field_accuracy = high_confidence_fields / total_fields if total_fields > 0 else 0
        
        # User correction metrics
        total_corrections = sum(self.user_corrections)
        correction_rate = total_corrections / len(self.results) if self.results else 0
        
        # Business value score (composite metric)
        business_value_score = self._compute_business_value_score()
        
        return {
            'processing_time_metrics': {
                'average_time': avg_processing_time,
                'time_std': processing_time_std,
                'min_time': np.min(self.processing_times),
                'max_time': np.max(self.processing_times)
            },
            'confidence_metrics': {
                'average_confidence': avg_confidence,
                'confidence_std': np.std(all_confidences) if all_confidences else 0
            },
            'field_extraction_metrics': {
                'total_fields_extracted': total_fields,
                'high_confidence_fields': high_confidence_fields,
                'field_accuracy': field_accuracy
            },
            'user_correction_metrics': {
                'total_corrections': total_corrections,
                'correction_rate': correction_rate
            },
            'business_value_score': business_value_score,
            'total_documents': len(self.results)
        }
    
    def _compute_business_value_score(self) -> float:
        """Compute composite business value score."""
        if not self.results:
            return 0.0
        
        # Factors contributing to business value
        factors = []
        
        # Processing speed (faster is better)
        avg_time = np.mean(self.processing_times)
        speed_score = max(0, 1 - (avg_time / 10))  # Normalize to 10 seconds
        factors.append(speed_score)
        
        # Confidence (higher is better)
        confidences = []
        for result in self.results:
            if 'confidence_summary' in result:
                confidences.append(result['confidence_summary'].get('average_confidence', 0))
        
        if confidences:
            confidence_score = np.mean(confidences)
            factors.append(confidence_score)
        
        # Field extraction rate (more fields is better)
        total_fields = sum(len(result.get('extracted_fields', [])) for result in self.results)
        field_score = min(1.0, total_fields / (len(self.results) * 10))  # Normalize to 10 fields per doc
        factors.append(field_score)
        
        # User correction rate (fewer corrections is better)
        correction_rate = sum(self.user_corrections) / len(self.results)
        correction_score = max(0, 1 - correction_rate)
        factors.append(correction_score)
        
        return np.mean(factors) if factors else 0.0
