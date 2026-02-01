import cv2
import numpy as np
import logging

class UIDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Dark Brown UI Box Range (Verified in research)
        self.lower_brown = np.array([5, 10, 30])
        self.upper_brown = np.array([30, 80, 120])
        
    def detect_question_box(self, image_np):
        """
        Detects the question box in the provided image (numpy array).
        Returns (x, y, w, h) or None if not found.
        """
        try:
            # 1. Convert to HSV
            if len(image_np.shape) == 3:
                hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            else:
                return None

            # 2. Mask
            mask = cv2.inRange(hsv, self.lower_brown, self.upper_brown)
            
            # 3. Morphological cleanup (close gaps)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # 4. Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 5. Filter for Question Box characteristics
            # - Should be in the top half (or middle) of the provided ROI
            # - Should be Wide
            height, width = image_np.shape[:2]
            best_box = None
            max_area = 0
            
            candidates = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                area = w * h
                
                # Heuristics:
                # - Width must be at least 20% of screen width (it's a wide banner)
                # - Height must be > 30px
                # - Aspect ratio should be > 2.0 (wide rectangle)
                if w > (width * 0.2) and h > 30 and aspect_ratio > 2.0:
                    candidates.append((x, y, w, h))
            
            if candidates:
                # Sort by Y position (find top-most box which is the Question)
                # Secondary sort by Area (descending) just in case
                candidates.sort(key=lambda b: (b[1], -b[2]*b[3]))
                
                # Verify the top candidate is not TOO close to the top edge (avoid status bars)
                # But typically Question is the first wide brown box.
                best_box = candidates[0]
                
                # DEBUG: Log if multiple candidates found
                if len(candidates) > 1:
                     self.logger.debug(f"SmartVision: Found {len(candidates)} candidates. Selected Top: {best_box}")
                
            return best_box
            
        except Exception as e:
            self.logger.error(f"Error in detect_question_box: {e}")
            return None
