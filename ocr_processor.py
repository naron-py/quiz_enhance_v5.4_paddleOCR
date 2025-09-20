import cv2
import numpy as np
import torch
import re
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
import time
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from concurrent.futures import ThreadPoolExecutor
from config_manager import ConfigManager

class OCRProcessor:
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the OCR processor with docTR"""
        self.logger = logging.getLogger(__name__)
        self.config = config if config is not None else ConfigManager().data
        self.scale_factor = self.config.get('image_scale_factor', 1.0)
        self.model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.model.eval()
            self.logger.info("Using GPU for OCR")
        else:
            self.model = self.model.cpu()
            self.model.eval()
            self.logger.info("Using CPU for OCR")
        self.inference_context = (
            torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        )
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
            with self.inference_context():
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        result = self.model([processed_image])
                else:
                    result = self.model([processed_image])
            
            # Extract text from all blocks
            text_parts = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        line_text = " ".join(word.value for word in line.words)
                        if line_text.strip():  # Only add non-empty lines
                            text_parts.append(line_text)
            
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
            # Batch OCR only on valid images
            with self.inference_context():
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        ocr_results = self.model(valid_images)
                else:
                    ocr_results = self.model(valid_images)

            if len(ocr_results.pages) != len(valid_names):
                self.logger.warning(
                    "Mismatch between OCR pages and valid regions: %d vs %d",
                    len(ocr_results.pages),
                    len(valid_names)
                )

            for name, page in zip(valid_names, ocr_results.pages):
                text_parts = []
                for block in page.blocks:
                    for line in block.lines:
                        line_text = " ".join(word.value for word in line.words)
                        if line_text.strip():
                            text_parts.append(line_text)
                full_text = ' '.join(text_parts)
                cleaned_text = self.clean_text(full_text)
                results[name] = cleaned_text
        except Exception as e:
            self.logger.error(f"Error during batch OCR: {str(e)}")
            return results

        if skipped_names:
            self.logger.warning(
                "Skipped OCR for regions without valid images: %s",
                ", ".join(skipped_names)
            )

        return results

    def export_detection_torchscript(self, export_path: str = "det_model.ts") -> Optional[str]:
        """Export the detection sub-model to TorchScript for investigation."""
        try:
            det_model = self.model.det_predictor.model
            det_model.eval()
            device = next(det_model.parameters()).device
            example = torch.randn(1, 3, 512, 512, device=device)
            scripted = torch.jit.trace(det_model, example)
            scripted.save(export_path)
            self.logger.info(f"Detection model exported to {export_path}")
            return export_path
        except Exception as e:
            self.logger.error(f"Failed to export detection model: {e}")
            return None

    def benchmark_detection(self, export_path: str = "det_model.ts", runs: int = 10) -> Dict[str, float]:
        """Benchmark inference speed between PyTorch and TorchScript detection models."""
        det_model = self.model.det_predictor.model
        device = next(det_model.parameters()).device
        dummy = torch.randn(1, 3, 512, 512, device=device)

        det_model.eval()
        for _ in range(3):
            det_model(dummy)
        start = time.time()
        for _ in range(runs):
            det_model(dummy)
        pytorch_time = (time.time() - start) / runs

        script = torch.jit.load(export_path, map_location=device)
        script.eval()
        for _ in range(3):
            script(dummy)
        start = time.time()
        for _ in range(runs):
            script(dummy)
        script_time = (time.time() - start) / runs

        self.logger.info(
            f"Detection benchmark - PyTorch: {pytorch_time:.4f}s, TorchScript: {script_time:.4f}s"
        )
        return {"pytorch": pytorch_time, "torchscript": script_time}