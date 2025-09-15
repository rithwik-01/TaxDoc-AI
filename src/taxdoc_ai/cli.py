#!/usr/bin/env python3
"""
Command-line interface for TaxDoc-AI.

This module provides a command-line interface for the tax document processing system.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional

from .pipeline.tax_document_processor import TaxDocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_single_document(args):
    """Process a single document."""
    processor = TaxDocumentProcessor(
        nvidia_api_key=args.nvidia_api_key,
        classifier_model_path=args.classifier_model,
        ner_model_path=args.ner_model
    )
    
    try:
        result = processor.process_document(args.image_path)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        sys.exit(1)

def process_batch(args):
    """Process multiple documents."""
    processor = TaxDocumentProcessor(
        nvidia_api_key=args.nvidia_api_key,
        classifier_model_path=args.classifier_model,
        ner_model_path=args.ner_model
    )
    
    # Read image paths from file or directory
    if args.input_file:
        with open(args.input_file, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
    else:
        # Process all images in directory
        input_dir = Path(args.input_dir)
        image_paths = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg"))
        image_paths = [str(p) for p in image_paths]
    
    try:
        results = processor.process_batch(image_paths)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(results, indent=2))
            
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        sys.exit(1)

def train_models(args):
    """Train models using the training scripts."""
    import subprocess
    
    scripts_dir = Path(__file__).parent.parent.parent / "scripts"
    
    if args.train_classifier:
        print("Training classifier...")
        cmd = [sys.executable, str(scripts_dir / "train_classifier.py")]
        if args.synthetic:
            cmd.append("--synthetic")
        if args.num_samples:
            cmd.extend(["--num_samples", str(args.num_samples)])
        if args.output_dir:
            cmd.extend(["--output_dir", args.output_dir])
        
        subprocess.run(cmd, check=True)
    
    if args.train_ner:
        print("Training NER model...")
        cmd = [sys.executable, str(scripts_dir / "train_ner.py")]
        if args.synthetic:
            cmd.append("--synthetic")
        if args.num_samples:
            cmd.extend(["--num_samples", str(args.num_samples)])
        if args.output_dir:
            cmd.extend(["--output_dir", args.output_dir])
        
        subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="TaxDoc-AI Command Line Interface")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process single document
    process_parser = subparsers.add_parser('process', help='Process a single document')
    process_parser.add_argument('image_path', help='Path to the document image')
    process_parser.add_argument('--output', '-o', help='Output file path')
    process_parser.add_argument('--nvidia_api_key', help='NVIDIA API key for OCR')
    process_parser.add_argument('--classifier_model', help='Path to trained classifier model')
    process_parser.add_argument('--ner_model', help='Path to trained NER model')
    process_parser.set_defaults(func=process_single_document)
    
    # Process batch
    batch_parser = subparsers.add_parser('batch', help='Process multiple documents')
    batch_parser.add_argument('--input_file', help='File containing list of image paths')
    batch_parser.add_argument('--input_dir', help='Directory containing images')
    batch_parser.add_argument('--output', '-o', help='Output file path')
    batch_parser.add_argument('--nvidia_api_key', help='NVIDIA API key for OCR')
    batch_parser.add_argument('--classifier_model', help='Path to trained classifier model')
    batch_parser.add_argument('--ner_model', help='Path to trained NER model')
    batch_parser.set_defaults(func=process_batch)
    
    # Train models
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--train_classifier', action='store_true', help='Train classifier')
    train_parser.add_argument('--train_ner', action='store_true', help='Train NER model')
    train_parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    train_parser.add_argument('--num_samples', type=int, help='Number of synthetic samples')
    train_parser.add_argument('--output_dir', help='Output directory for models')
    train_parser.set_defaults(func=train_models)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
