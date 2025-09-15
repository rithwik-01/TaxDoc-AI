"""
OCR processors for extracting text from tax document images.

This module provides multiple OCR implementations:
1. NVIDIA OCR (via API) - High accuracy, requires API key
2. PaddleOCR (open source) - Good accuracy, no API required
"""

import base64
import requests
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

class BaseOCRProcessor(ABC):
    """Abstract base class for OCR processors."""
    
    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Extract text from image file."""
        pass
    
    @abstractmethod
    def extract_text_with_coordinates(self, image_path: str) -> List[Dict]:
        """Extract text with bounding box coordinates."""
        pass

class NVIDIAOCRProcessor(BaseOCRProcessor):
    """
    NVIDIA OCR processor using NVIDIA NIM (Neural Inference Microservices).
    
    This processor uses NVIDIA's cloud-based OCR API for high-accuracy text extraction.
    Requires an NVIDIA API key for authentication.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://ai.api.nvidia.com/v1/cv/nvidia/ocr"):
        """
        Initialize NVIDIA OCR processor.
        
        Args:
            api_key: NVIDIA API key for authentication
            base_url: Base URL for NVIDIA OCR API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from image using NVIDIA OCR API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as string
        """
        try:
            # Read and encode image
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
            
            # Prepare API request
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Extract all text from this tax document: <img src=\"data:image/jpeg;base64,{image_b64}\" />"
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.1
            }
            
            # Make API request
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            # Extract text from response
            result = response.json()
            extracted_text = result["choices"][0]["message"]["content"]
            
            logger.info(f"Successfully extracted text using NVIDIA OCR: {len(extracted_text)} characters")
            return extracted_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"NVIDIA OCR API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error extracting text with NVIDIA OCR: {e}")
            raise
    
    def extract_text_with_coordinates(self, image_path: str) -> List[Dict]:
        """
        Extract text with coordinates using NVIDIA OCR API.
        
        Note: NVIDIA API doesn't provide coordinate information in the current implementation.
        This method returns text with estimated confidence scores.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries with text and confidence information
        """
        try:
            text = self.extract_text(image_path)
            
            # Split text into lines and create coordinate-like structure
            lines = text.split('\n')
            results = []
            
            for i, line in enumerate(lines):
                if line.strip():
                    results.append({
                        'text': line.strip(),
                        'confidence': 0.9,  # High confidence for NVIDIA OCR
                        'bbox': [[0, i*20], [len(line)*10, i*20], [len(line)*10, (i+1)*20], [0, (i+1)*20]]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting text with coordinates using NVIDIA OCR: {e}")
            raise

class PaddleOCRProcessor(BaseOCRProcessor):
    """
    PaddleOCR processor for open-source OCR functionality.
    
    This processor uses PaddleOCR library for text extraction with bounding box coordinates.
    No API key required, runs locally.
    """
    
    def __init__(self, use_angle_cls: bool = True, lang: str = 'en'):
        """
        Initialize PaddleOCR processor.
        
        Args:
            use_angle_cls: Whether to use angle classification
            lang: Language for OCR (default: English)
        """
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=False)
            logger.info("PaddleOCR initialized successfully")
        except ImportError:
            logger.error("PaddleOCR not installed. Please install with: pip install paddleocr")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise
    
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from image using PaddleOCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as string
        """
        try:
            # Run OCR
            result = self.ocr.ocr(image_path, cls=True)
            
            # Extract text from results
            extracted_text = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]  # Extract text from (text, confidence) tuple
                        extracted_text.append(text)
            
            full_text = ' '.join(extracted_text)
            logger.info(f"Successfully extracted text using PaddleOCR: {len(full_text)} characters")
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text with PaddleOCR: {e}")
            raise
    
    def extract_text_with_coordinates(self, image_path: str) -> List[Dict]:
        """
        Extract text with bounding box coordinates using PaddleOCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries with text, confidence, and bounding box information
        """
        try:
            # Run OCR
            result = self.ocr.ocr(image_path, cls=True)
            
            extracted_data = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        bbox, (text, confidence) = line
                        extracted_data.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': bbox
                        })
            
            logger.info(f"Successfully extracted {len(extracted_data)} text regions using PaddleOCR")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting text with coordinates using PaddleOCR: {e}")
            raise

class OCRProcessorFactory:
    """Factory class for creating OCR processors."""
    
    @staticmethod
    def create_processor(processor_type: str = "paddle", **kwargs) -> BaseOCRProcessor:
        """
        Create an OCR processor instance.
        
        Args:
            processor_type: Type of processor ("nvidia" or "paddle")
            **kwargs: Additional arguments for processor initialization
            
        Returns:
            OCR processor instance
        """
        if processor_type.lower() == "nvidia":
            if "api_key" not in kwargs:
                raise ValueError("NVIDIA OCR processor requires 'api_key' parameter")
            return NVIDIAOCRProcessor(**kwargs)
        elif processor_type.lower() == "paddle":
            return PaddleOCRProcessor(**kwargs)
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")
    
    @staticmethod
    def create_with_fallback(nvidia_api_key: Optional[str] = None, **kwargs) -> BaseOCRProcessor:
        """
        Create OCR processor with fallback logic.
        
        Args:
            nvidia_api_key: NVIDIA API key (if available)
            **kwargs: Additional arguments for processor initialization
            
        Returns:
            OCR processor instance (NVIDIA if API key available, otherwise PaddleOCR)
        """
        if nvidia_api_key:
            try:
                return NVIDIAOCRProcessor(api_key=nvidia_api_key, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to initialize NVIDIA OCR, falling back to PaddleOCR: {e}")
        
        return PaddleOCRProcessor(**kwargs)

def preprocess_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> str:
    """
    Preprocess image for better OCR results.
    
    Args:
        image_path: Path to the input image
        target_size: Target size for resizing (width, height)
        
    Returns:
        Path to the preprocessed image
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Resize if target size specified
        if target_size:
            thresh = cv2.resize(thresh, target_size, interpolation=cv2.INTER_CUBIC)
        
        # Save preprocessed image
        preprocessed_path = image_path.replace('.', '_preprocessed.')
        cv2.imwrite(preprocessed_path, thresh)
        
        logger.info(f"Image preprocessed and saved to: {preprocessed_path}")
        return preprocessed_path
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise
