"""
Comprehensive evaluator for the tax document processing system.

This module provides end-to-end evaluation capabilities for all components
of the tax document processing pipeline.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time
import pandas as pd

from .metrics import ClassificationMetrics, NERMetrics, PipelineMetrics
from ..pipeline.tax_document_processor import TaxDocumentProcessor
from ..classifiers.document_classifier import DOCUMENT_TYPES
from ..ner.field_extractor import TAX_ENTITY_TYPES

logger = logging.getLogger(__name__)

class TaxDocumentEvaluator:
    """
    Comprehensive evaluator for tax document processing system.
    
    This class provides evaluation capabilities for:
    1. Document classification
    2. Named entity recognition
    3. End-to-end pipeline performance
    """
    
    def __init__(self, processor: TaxDocumentProcessor):
        """
        Initialize the evaluator.
        
        Args:
            processor: TaxDocumentProcessor instance to evaluate
        """
        self.processor = processor
        
        # Initialize metrics collectors
        self.classification_metrics = ClassificationMetrics(DOCUMENT_TYPES)
        self.ner_metrics = NERMetrics(TAX_ENTITY_TYPES)
        self.pipeline_metrics = PipelineMetrics()
        
        # Evaluation results storage
        self.evaluation_results = {}
    
    def evaluate_classification(
        self, 
        test_data: List[Dict[str, Any]], 
        save_results: bool = True,
        output_dir: str = "./evaluation_results"
    ) -> Dict[str, Any]:
        """
        Evaluate document classification performance.
        
        Args:
            test_data: List of test samples with 'text' and 'true_label' keys
            save_results: Whether to save results to file
            output_dir: Directory to save results
            
        Returns:
            Classification evaluation results
        """
        logger.info(f"Evaluating classification on {len(test_data)} samples")
        
        self.classification_metrics.reset()
        
        for i, sample in enumerate(test_data):
            try:
                start_time = time.time()
                
                # Get prediction
                predicted_type = self.processor.classifier.predict(sample['text'])
                confidence = self.processor.classifier.get_confidence_score(sample['text'])
                
                processing_time = time.time() - start_time
                
                # Update metrics
                self.classification_metrics.update(
                    prediction=predicted_type,
                    true_label=sample['true_label'],
                    confidence=confidence,
                    processing_time=processing_time
                )
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_data)} samples")
                    
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
        
        # Compute metrics
        results = self.classification_metrics.compute_metrics()
        
        # Save results
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "classification_evaluation.json"), 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save detailed results
            self._save_classification_details(test_data, output_dir)
        
        self.evaluation_results['classification'] = results
        logger.info(f"Classification evaluation completed. Accuracy: {results.get('accuracy', 0):.4f}")
        
        return results
    
    def evaluate_ner(
        self, 
        test_data: List[Dict[str, Any]], 
        save_results: bool = True,
        output_dir: str = "./evaluation_results"
    ) -> Dict[str, Any]:
        """
        Evaluate named entity recognition performance.
        
        Args:
            test_data: List of test samples with 'text' and 'true_entities' keys
            save_results: Whether to save results to file
            output_dir: Directory to save results
            
        Returns:
            NER evaluation results
        """
        logger.info(f"Evaluating NER on {len(test_data)} samples")
        
        self.ner_metrics.reset()
        
        for i, sample in enumerate(test_data):
            try:
                # Get NER predictions
                predicted_entities = self.processor.ner_extractor.extract_tax_entities(sample['text'])
                
                # Calculate average confidence
                all_confidences = []
                for entity_list in predicted_entities.values():
                    all_confidences.extend([e['confidence'] for e in entity_list])
                avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
                
                # Update metrics
                self.ner_metrics.update(
                    predicted_entities=predicted_entities,
                    true_entities=sample['true_entities'],
                    confidence=avg_confidence
                )
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_data)} samples")
                    
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
        
        # Compute metrics
        results = self.ner_metrics.compute_metrics()
        
        # Save results
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "ner_evaluation.json"), 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save detailed results
            self._save_ner_details(test_data, output_dir)
        
        self.evaluation_results['ner'] = results
        logger.info(f"NER evaluation completed. F1: {results.get('overall_metrics', {}).get('f1', 0):.4f}")
        
        return results
    
    def evaluate_pipeline(
        self, 
        test_data: List[Dict[str, Any]], 
        save_results: bool = True,
        output_dir: str = "./evaluation_results"
    ) -> Dict[str, Any]:
        """
        Evaluate end-to-end pipeline performance.
        
        Args:
            test_data: List of test samples with 'image_path' and optional 'user_corrections' keys
            save_results: Whether to save results to file
            output_dir: Directory to save results
            
        Returns:
            Pipeline evaluation results
        """
        logger.info(f"Evaluating pipeline on {len(test_data)} samples")
        
        self.pipeline_metrics.reset()
        
        for i, sample in enumerate(test_data):
            try:
                start_time = time.time()
                
                # Process document
                result = self.processor.process_document(sample['image_path'])
                
                processing_time = time.time() - start_time
                
                # Update metrics
                self.pipeline_metrics.update(
                    result=result,
                    processing_time=processing_time,
                    user_corrections=sample.get('user_corrections', 0)
                )
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_data)} documents")
                    
            except Exception as e:
                logger.error(f"Error processing document {i}: {e}")
                continue
        
        # Compute metrics
        results = self.pipeline_metrics.compute_metrics()
        
        # Save results
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "pipeline_evaluation.json"), 'w') as f:
                json.dump(results, f, indent=2)
        
        self.evaluation_results['pipeline'] = results
        logger.info(f"Pipeline evaluation completed. Business value score: {results.get('business_value_score', 0):.4f}")
        
        return results
    
    def evaluate_all(
        self, 
        test_data: Dict[str, List[Dict[str, Any]]], 
        save_results: bool = True,
        output_dir: str = "./evaluation_results"
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on all components.
        
        Args:
            test_data: Dictionary with 'classification', 'ner', and 'pipeline' keys
            save_results: Whether to save results to file
            output_dir: Directory to save results
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting comprehensive evaluation")
        
        results = {}
        
        # Evaluate classification
        if 'classification' in test_data:
            results['classification'] = self.evaluate_classification(
                test_data['classification'], save_results, output_dir
            )
        
        # Evaluate NER
        if 'ner' in test_data:
            results['ner'] = self.evaluate_ner(
                test_data['ner'], save_results, output_dir
            )
        
        # Evaluate pipeline
        if 'pipeline' in test_data:
            results['pipeline'] = self.evaluate_pipeline(
                test_data['pipeline'], save_results, output_dir
            )
        
        # Generate summary report
        summary = self._generate_summary_report(results)
        results['summary'] = summary
        
        # Save complete results
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "complete_evaluation.json"), 'w') as f:
                json.dump(results, f, indent=2)
            
            # Generate HTML report
            self._generate_html_report(results, output_dir)
        
        logger.info("Comprehensive evaluation completed")
        return results
    
    def _save_classification_details(self, test_data: List[Dict[str, Any]], output_dir: str):
        """Save detailed classification results."""
        details = []
        
        for i, sample in enumerate(test_data):
            try:
                predicted_type = self.processor.classifier.predict(sample['text'])
                confidence = self.processor.classifier.get_confidence_score(sample['text'])
                
                details.append({
                    'sample_id': i,
                    'text': sample['text'][:200] + "..." if len(sample['text']) > 200 else sample['text'],
                    'true_label': sample['true_label'],
                    'predicted_label': predicted_type,
                    'confidence': confidence,
                    'correct': predicted_type == sample['true_label']
                })
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
        
        # Save as CSV
        df = pd.DataFrame(details)
        df.to_csv(os.path.join(output_dir, "classification_details.csv"), index=False)
    
    def _save_ner_details(self, test_data: List[Dict[str, Any]], output_dir: str):
        """Save detailed NER results."""
        details = []
        
        for i, sample in enumerate(test_data):
            try:
                predicted_entities = self.processor.ner_extractor.extract_tax_entities(sample['text'])
                
                details.append({
                    'sample_id': i,
                    'text': sample['text'][:200] + "..." if len(sample['text']) > 200 else sample['text'],
                    'true_entities': sample['true_entities'],
                    'predicted_entities': predicted_entities
                })
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
        
        # Save as JSON
        with open(os.path.join(output_dir, "ner_details.json"), 'w') as f:
            json.dump(details, f, indent=2)
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary report of all evaluation results."""
        summary = {
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_performance': {},
            'component_performance': {}
        }
        
        # Classification summary
        if 'classification' in results:
            cls_results = results['classification']
            summary['component_performance']['classification'] = {
                'accuracy': cls_results.get('accuracy', 0),
                'f1_score': cls_results.get('f1', 0),
                'average_confidence': cls_results.get('confidence_metrics', {}).get('average_confidence', 0)
            }
        
        # NER summary
        if 'ner' in results:
            ner_results = results['ner']
            summary['component_performance']['ner'] = {
                'f1_score': ner_results.get('overall_metrics', {}).get('f1', 0),
                'precision': ner_results.get('overall_metrics', {}).get('precision', 0),
                'recall': ner_results.get('overall_metrics', {}).get('recall', 0),
                'exact_match_accuracy': ner_results.get('match_metrics', {}).get('exact_match_accuracy', 0)
            }
        
        # Pipeline summary
        if 'pipeline' in results:
            pipeline_results = results['pipeline']
            summary['component_performance']['pipeline'] = {
                'business_value_score': pipeline_results.get('business_value_score', 0),
                'average_processing_time': pipeline_results.get('processing_time_metrics', {}).get('average_time', 0),
                'field_accuracy': pipeline_results.get('field_extraction_metrics', {}).get('field_accuracy', 0)
            }
        
        # Overall performance score
        component_scores = []
        for component, metrics in summary['component_performance'].items():
            if component == 'classification':
                component_scores.append(metrics.get('accuracy', 0))
            elif component == 'ner':
                component_scores.append(metrics.get('f1_score', 0))
            elif component == 'pipeline':
                component_scores.append(metrics.get('business_value_score', 0))
        
        summary['overall_performance']['average_score'] = sum(component_scores) / len(component_scores) if component_scores else 0
        summary['overall_performance']['component_count'] = len(component_scores)
        
        return summary
    
    def _generate_html_report(self, results: Dict[str, Any], output_dir: str):
        """Generate HTML evaluation report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TaxDoc-AI Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
                .good {{ background-color: #d4edda; }}
                .warning {{ background-color: #fff3cd; }}
                .error {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>TaxDoc-AI Evaluation Report</h1>
                <p>Generated on: {results.get('summary', {}).get('evaluation_timestamp', 'Unknown')}</p>
            </div>
        """
        
        # Add summary section
        if 'summary' in results:
            summary = results['summary']
            html_content += f"""
            <div class="section">
                <h2>Overall Performance</h2>
                <div class="metric {'good' if summary.get('overall_performance', {}).get('average_score', 0) > 0.8 else 'warning' if summary.get('overall_performance', {}).get('average_score', 0) > 0.6 else 'error'}">
                    Average Score: {summary.get('overall_performance', {}).get('average_score', 0):.3f}
                </div>
            </div>
            """
        
        # Add component sections
        for component, metrics in results.items():
            if component == 'summary':
                continue
                
            html_content += f"""
            <div class="section">
                <h2>{component.title()} Performance</h2>
            """
            
            if component == 'classification':
                html_content += f"""
                <div class="metric {'good' if metrics.get('accuracy', 0) > 0.9 else 'warning' if metrics.get('accuracy', 0) > 0.7 else 'error'}">
                    Accuracy: {metrics.get('accuracy', 0):.3f}
                </div>
                <div class="metric">
                    F1 Score: {metrics.get('f1', 0):.3f}
                </div>
                """
            
            elif component == 'ner':
                overall = metrics.get('overall_metrics', {})
                html_content += f"""
                <div class="metric {'good' if overall.get('f1', 0) > 0.8 else 'warning' if overall.get('f1', 0) > 0.6 else 'error'}">
                    F1 Score: {overall.get('f1', 0):.3f}
                </div>
                <div class="metric">
                    Precision: {overall.get('precision', 0):.3f}
                </div>
                <div class="metric">
                    Recall: {overall.get('recall', 0):.3f}
                </div>
                """
            
            elif component == 'pipeline':
                html_content += f"""
                <div class="metric {'good' if metrics.get('business_value_score', 0) > 0.8 else 'warning' if metrics.get('business_value_score', 0) > 0.6 else 'error'}">
                    Business Value Score: {metrics.get('business_value_score', 0):.3f}
                </div>
                <div class="metric">
                    Average Processing Time: {metrics.get('processing_time_metrics', {}).get('average_time', 0):.2f}s
                </div>
                """
            
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(os.path.join(output_dir, "evaluation_report.html"), 'w') as f:
            f.write(html_content)
