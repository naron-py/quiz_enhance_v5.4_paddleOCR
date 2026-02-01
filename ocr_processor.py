import sys
try:
    from paddleocr import PaddleOCR
    import paddle
except ImportError as e:
    import logging
    logging.error(f"Failed to import paddleocr. Executable: {sys.executable}")
    logging.error(f"Sys Path: {sys.path}")
    raise e

import os
import cv2
import numpy as np
import re
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
import time
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from config_manager import ConfigManager

class OCRProcessor:
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the OCR processor with docTR"""
        self.logger = logging.getLogger(__name__)
        self.config = config if config is not None else ConfigManager().data
        self.scale_factor = self.config.get('image_scale_factor', 1.0)
        self.use_gpu = self.config.get('require_cuda', False) or self.config.get('use_gpu', True)
        
        try:
            # Control PaddleOCR logging based on config (must be set BEFORE initialization)
            show_paddleocr_logs = self.config.get('show_paddleocr_debug_logs', False)
            if not show_paddleocr_logs:
                # Suppress PaddleOCR debug logs
                import logging as ppocr_logging
                ppocr_logging.getLogger('ppocr').setLevel(ppocr_logging.WARNING)
            
            # Initialize PaddleOCR
            # use_angle_cls=True allows detecting text at angles
            # lang='en' for English
            # Set device explicitly via paddle
            # Bypass model source connectivity check to avoid torch import issues
            os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
            
            if self.use_gpu:
                paddle.device.set_device('gpu')
            else:
                paddle.device.set_device('cpu')
                
            self.model = PaddleOCR(use_angle_cls=True, lang='en')
            
            self.logger.info(f"PaddleOCR initialized (GPU={self.use_gpu})")
        except Exception as e:
            self.logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

        self.debug_dir = Path('debug_images')
        self.debug_dir.mkdir(exist_ok=True)
        
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        """Enhanced image preprocessing for better text recognition"""
        try:
            if image is None:
                self.logger.warning("Received None image for preprocessing; skipping.")
                return None

            if isinstance(image, str) and not image:
                self.logger.warning("Received empty image path for preprocessing; skipping.")
                return None

            if isinstance(image, np.ndarray) and image.size == 0:
                self.logger.warning("Received empty numpy array for preprocessing; skipping.")
                return None

            if isinstance(image, Image.Image) and (image.size[0] == 0 or image.size[1] == 0):
                self.logger.warning("Received empty PIL image for preprocessing; skipping.")
                return None

            # Convert input to numpy array
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
                image_np = np.array(image)
            elif isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image

            if image_np.size == 0:
                self.logger.warning("Received image with no data after conversion; skipping.")
                return None

            # Convert to RGB if needed
            if len(image_np.shape) == 2:  # Grayscale
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 3:  # Already RGB
                pass
            else:
                raise ValueError(f"Unexpected image shape: {image_np.shape}")

            # Optionally resize to limit dimensions
            if self.scale_factor and self.scale_factor != 1.0:
                new_size = (
                    int(image_np.shape[1] * self.scale_factor),
                    int(image_np.shape[0] * self.scale_factor)
                )
                interpolation = cv2.INTER_AREA if self.scale_factor < 1.0 else cv2.INTER_LINEAR
                image_np = cv2.resize(image_np, new_size, interpolation=interpolation)

            # Enhance contrast using CLAHE
            lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            return None

    def clean_text(self, text: str) -> str:
        """Clean and normalize OCR text output"""
        if not text:
            return ""
            
        # Remove unwanted characters while preserving essential punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\-.,!?\'"]', '', text)
        
        # Fix spacing issues
        text = re.sub(r'(?<![\'\s])([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase
        text = re.sub(r'\s+', ' ', text)  # Fix multiple spaces
        text = re.sub(r'\s*-\s*', '-', text)  # Fix spacing around hyphens
        text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)  # Fix spacing after punctuation
        
        return text.strip()

    def recognize_text(self, image: Union[str, np.ndarray, Image.Image], min_confidence: float = 0.5) -> str:
        """Perform OCR on the image and return recognized text"""
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)

            if processed_image is None:
                self.logger.warning("Skipping OCR because preprocessing returned no data.")
                return ""

            # Perform OCR using docTR
            # Perform OCR using PaddleOCR
            # PaddleOCR expects path or numpy array
            result = self.model.ocr(processed_image, cls=True)

            # Extract text from result
            # result structure: [ [ [ [x1,y1],...], (text, conf) ], ... ]
            # Note: result can be None or empty list if no text found
            
            text_parts = []
            if result and result[0]:
                for line in result[0]:
                    # line[1] contains (text, confidence)
                    text = line[1][0]
                    if text.strip():
                        text_parts.append(text)
            
            # Join and clean the text
            full_text = ' '.join(text_parts)
            cleaned_text = self.clean_text(full_text)
            
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"Error in OCR process: {str(e)}")
            return ""

    def process_quiz_regions(self, regions: Dict[str, np.ndarray]) -> Dict[str, str]:
        """Process multiple quiz regions (question and answers) using batch OCR for maximum GPU speed"""
        # Prepare images in order
        region_names = list(regions.keys())
        
        def _preprocess(item: Tuple[str, np.ndarray]) -> Tuple[str, Optional[np.ndarray]]:
            name, region_image = item
            return name, self.preprocess_image(region_image)

        with ThreadPoolExecutor() as executor:
            processed_pairs = list(executor.map(_preprocess, regions.items()))

        skipped_names: List[str] = []
        valid_names: List[str] = []
        valid_images: List[np.ndarray] = []
        results = {name: "" for name in region_names}

        for name, processed_image in processed_pairs:
            if processed_image is None or (isinstance(processed_image, np.ndarray) and processed_image.size == 0):
                skipped_names.append(name)
            else:
                valid_names.append(name)
                valid_images.append(processed_image)

        if len(skipped_names) == len(region_names):
            if skipped_names:
                self.logger.warning(
                    "All regions skipped during preprocessing: %s",
                    ", ".join(skipped_names)
                )
            return results

        try:
            # Prepare images list for batch processing if supported, 
            # but PaddleOCR standard API processes one by one mostly unless using specific batch APIs which are less standard.
            # However, we can just loop them. For true parallel/batch, we'd need to check Paddle docs deeply.
            # But the requirement is speed. Sequential processing of pre-loaded numpy arrays is reasonably fast on GPU.
            # Let's try simple sequential first, as PaddleOCR internal batching isn't straightforward in the high-level API.
            
            for name, img in zip(valid_names, valid_images):
                try:
                    res = self.model.ocr(img, cls=True)
                    parts = []
                    if res and res[0]:
                        for line in res[0]:
                            parts.append(line[1][0])
                    full_text = ' '.join(parts)
                    cleaned_text = self.clean_text(full_text)
                    results[name] = cleaned_text
                except Exception as e:
                    self.logger.error(f"Error OCRing region {name}: {e}")

        except Exception as e:
            self.logger.error(f"Error during batch OCR: {str(e)}")
            return results

        if skipped_names:
            self.logger.warning(
                "Skipped OCR for regions without valid images: %s",
                ", ".join(skipped_names)
            )

        return results


