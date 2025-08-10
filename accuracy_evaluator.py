import cv2
import numpy as np
import pandas as pd
import json
import os
import re
import difflib
import logging
import sys
import gc  # For garbage collection
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from config_manager import ConfigManager

# --- Removed sklearn imports as TF-IDF is no longer used ---
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# --- Removed end ---

# --- OCR Processor ---
# Assuming ocr_processor.py exists in the same directory or PYTHONPATH
try:
    from ocr_processor import OCRProcessor
except ImportError:
    print("Error: Could not import OCRProcessor. Make sure ocr_processor.py is accessible.")
    sys.exit(1)
# --- OCR Processor end ---

# --- Optional Reporting Libraries ---
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    COLORAMA = True
except ImportError:
    COLORAMA = False
    class Dummy:
        def __getattr__(self, name):
            return '' # Return empty string for any attribute access
    Fore = Style = Dummy()
    Fore.RESET = '' # Specifically handle RESET_ALL

try:
    from prettytable import PrettyTable
    HAS_PRETTYTABLE = True
except ImportError:
    HAS_PRETTYTABLE = False

try:
    import editdistance
    HAS_EDITDISTANCE = True
except ImportError:
    HAS_EDITDISTANCE = False
    print("Warning: 'editdistance' library not found. CER/WER calculations will be skipped.")
    print("Install it using: pip install editdistance")
# --- Optional Reporting Libraries end ---

# --- Path and Logging Setup ---
PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIG_FILE = PROJECT_ROOT / 'config.json'
log_file = PROJECT_ROOT / 'accuracy_evaluator.log'

config_manager = ConfigManager(str(CONFIG_FILE))
config = config_manager.data

log_level_name = config.get('log_level', 'INFO').upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print logs to console
    ]
)
logger = logging.getLogger(__name__)
# --- Path and Logging Setup end ---

# --- Global Variables & Constants ---
ALL_CAPTURES_DIR = PROJECT_ROOT / "all_captures"
REPORTS_DIR = PROJECT_ROOT / "accuracy_reports"
# --- Changed from DATABASE_FILE to GROUND_TRUTH_FILE ---
GROUND_TRUTH_FILE = PROJECT_ROOT / 'compare.json' # Using compare.json as ground truth
# --- Changed end ---


# --- Filter Selected Pattern Function (Added from terminal_app.py) ---
def filter_selected_pattern(text):
    """Filter out '[number] selected' pattern from text"""
    if not text:
        return text
    
    # Pattern: matches text ending with 1 or more digits followed by "selected" (with optional space)
    pattern = r'\s*\d+\s*selected\s*$'
    return re.sub(pattern, '', text).strip()
# --- Filter Selected Pattern Function end ---

# --- CER/WER Utility Functions (Unchanged) ---
def compute_cer(ref: str, hyp: str) -> Optional[float]:
    """Compute Character Error Rate (CER) between reference and hypothesis."""
    if not HAS_EDITDISTANCE: return None
    ref = str(ref) if ref is not None else ''
    hyp = str(hyp) if hyp is not None else ''
    if not ref and not hyp: return 0.0
    if not ref: return 1.0 # Entire hypothesis is insertion
    try:
        # Cast distance to float before division
        distance = float(editdistance.eval(ref, hyp))
        return distance / len(ref)
    except ZeroDivisionError:
        return 1.0 # Reference is empty, hypothesis is not
    except Exception as e:
        logger.error(f"Error computing CER for ref='{ref}', hyp='{hyp}': {e}")
        return None # Return None on other errors

def compute_wer(ref: str, hyp: str) -> Optional[float]:
    """Compute Word Error Rate (WER) between reference and hypothesis."""
    if not HAS_EDITDISTANCE: return None
    ref = str(ref) if ref is not None else ''
    hyp = str(hyp) if hyp is not None else ''
    if not ref and not hyp: return 0.0
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
         # If ref is empty (no words), WER is 1.0 if hyp has words, 0.0 if both empty
         return 1.0 if hyp_words else 0.0
    try:
        # Cast distance to float before division
        distance = float(editdistance.eval(ref_words, hyp_words))
        return distance / len(ref_words)
    except ZeroDivisionError:
         # This case is handled by the check for `not ref_words` above
         # but added defensively. If ref_words is empty, len is 0.
        return 1.0 if hyp_words else 0.0
    except Exception as e:
        logger.error(f"Error computing WER for ref='{ref}', hyp='{hyp}': {e}")
        return None # Return None on other errors
# --- CER/WER Utility Functions end ---


# --- Accuracy Evaluation & Reporting (Modified) ---
def evaluate_accuracy(results: List[Dict[str, Any]], project_root: Path):
    """
    Analyzes the results from processing images and prints/saves a comparison report
    against the ground truth JSON file.

    Args:
        results: A list of dictionaries, where each dict contains details
                 about the processing of one image. Expected keys include:
                 'image_filename', 'ground_truth_found' (bool),
                 'gt_question', 'ocr_question',
                 'gt_answer_A', 'ocr_answer_A', ... (for B, C, D),
                 'error_message' (optional).
        project_root: The root path of the project.
    """
    logger.info("Starting comparison evaluation...")
    if not results:
        logger.warning("No results provided for evaluation.")
        print("No results to evaluate.")
        return

    try:
        # --- Calculate CER/WER for each result --- 
        total_images = len(results)
        images_with_errors = 0
        images_with_ground_truth = 0

        for res in results:
            if res.get('error_message'):
                images_with_errors += 1
                # Don't skip CER/WER if ground truth was found but OCR failed partially

            if res.get('ground_truth_found', False):
                images_with_ground_truth += 1
                
                # --- Question CER/WER ---
                gt_q = res.get('gt_question', '')
                ocr_q = res.get('ocr_question', '')
                res['question_CER'] = compute_cer(gt_q, ocr_q)
                res['question_WER'] = compute_wer(gt_q, ocr_q)

                # --- Answer CER/WER (per choice) ---
                for choice in ['A', 'B', 'C', 'D']:
                    gt_ans = res.get(f'gt_answer_{choice}', '')
                    ocr_ans = res.get(f'ocr_answer_{choice}', '')
                    res[f'answer_{choice}_CER'] = compute_cer(gt_ans, ocr_ans)
                    res[f'answer_{choice}_WER'] = compute_wer(gt_ans, ocr_ans)
            else:
                # Mark CER/WER as not applicable if ground truth wasn't found
                res['question_CER'] = None
                res['question_WER'] = None
                for choice in ['A', 'B', 'C', 'D']:
                    res[f'answer_{choice}_CER'] = None
                    res[f'answer_{choice}_WER'] = None

        # --- Aggregate Average CER/WER --- 
        # Filter results where ground truth was found AND CER/WER is a valid number
        valid_q_cer = [r['question_CER'] for r in results if r.get('ground_truth_found') and isinstance(r.get('question_CER'), float)]
        valid_q_wer = [r['question_WER'] for r in results if r.get('ground_truth_found') and isinstance(r.get('question_WER'), float)]
        avg_q_cer = sum(valid_q_cer) / len(valid_q_cer) if valid_q_cer else 0.0
        avg_q_wer = sum(valid_q_wer) / len(valid_q_wer) if valid_q_wer else 0.0

        avg_ans_cer = {}
        avg_ans_wer = {}
        valid_ans_cer_counts = {}
        valid_ans_wer_counts = {}
        for choice in ['A', 'B', 'C', 'D']:
             valid_cer = [r[f'answer_{choice}_CER'] for r in results if r.get('ground_truth_found') and isinstance(r.get(f'answer_{choice}_CER'), float)]
             valid_wer = [r[f'answer_{choice}_WER'] for r in results if r.get('ground_truth_found') and isinstance(r.get(f'answer_{choice}_WER'), float)]
             avg_ans_cer[choice] = sum(valid_cer) / len(valid_cer) if valid_cer else 0.0
             avg_ans_wer[choice] = sum(valid_wer) / len(valid_wer) if valid_wer else 0.0
             valid_ans_cer_counts[choice] = len(valid_cer)
             valid_ans_wer_counts[choice] = len(valid_wer)

        # --- Console Output --- 
        def color(val, c):
             return f"{c}{val}{Style.RESET_ALL}" if COLORAMA else str(val)

        print("\n================ OCR COMPARISON REPORT ================\n")
        print(color("--- Summary ---", Fore.CYAN))
        print(f"Total Images Processed: {color(total_images, Fore.YELLOW)}")
        print(f"Images with Processing Errors: {color(images_with_errors, Fore.RED if images_with_errors > 0 else Fore.GREEN)}")
        print(f"Images Found in Ground Truth ({GROUND_TRUTH_FILE.name}): {color(images_with_ground_truth, Fore.YELLOW)}")
        
        if HAS_EDITDISTANCE:
            print("\n" + color("--- Average Error Rate Metrics (on images with ground truth) ---", Fore.CYAN))
            print(f"  Average Question CER: {color(f'{avg_q_cer:.4f}', Fore.MAGENTA)} ({len(valid_q_cer)} images)")
            print(f"  Average Question WER: {color(f'{avg_q_wer:.4f}', Fore.MAGENTA)} ({len(valid_q_wer)} images)")
            for choice in ['A', 'B', 'C', 'D']:
                 print(f"  Average Answer '{choice}' CER: {color(f'{avg_ans_cer[choice]:.4f}', Fore.MAGENTA)} ({valid_ans_cer_counts[choice]} images)")
                 print(f"  Average Answer '{choice}' WER: {color(f'{avg_ans_wer[choice]:.4f}', Fore.MAGENTA)} ({valid_ans_wer_counts[choice]} images)")
        else:
             print("\n" + color("--- Error Rate Metrics (Skipped) ---", Fore.YELLOW))
             print("Install 'editdistance' library to calculate CER/WER.")

        print("-------------------------------------------------------------\n")

        # --- Per-image breakdown --- 
        print(color("--- Per-Image Breakdown ---", Fore.CYAN)) 
        max_img_len = max(len(res.get('image_filename', '')) for res in results) if results else 30
        img_col_width = max(30, min(max_img_len, 50))
        err_rate_col_width = 7 # Width for individual CER/WER columns 

        # Define headers with separate CER/WER columns
        headers = ["#", "Image", "Q CER", "Q WER", "A CER", "A WER", "B CER", "B WER", "C CER", "C WER", "D CER", "D WER", "Error"]
        # Define alignment: Left for #, Image, Error; Right for CER/WER
        col_align = ["<", "<"] + [">"] * 10 + ["<"] 
        # Define widths
        col_widths = [3, img_col_width] + [err_rate_col_width] * 10 + [30] 

        # Create header format string for manual table fallback
        header_fmt_parts = []
        for i, h in enumerate(headers):
            header_fmt_parts.append(f"{{:{col_align[i]}{col_widths[i]}}}")
        header_format_string = " ".join(header_fmt_parts)
        row_format_string = header_format_string # Use same for rows
        separator_length = sum(col_widths) + len(col_widths) - 1 # Approx separator length

        if HAS_PRETTYTABLE:
            table = PrettyTable()
            table.field_names = headers
            # Set alignment based on col_align
            for i, h in enumerate(headers):
                 align_char = 'l' if col_align[i] == '<' else ('r' if col_align[i] == '>' else 'c')
                 table.align[h] = align_char
            
            for idx, res in enumerate(results, 1):
                img_name = res.get('image_filename', '')
                err_msg = res.get('error_message', '')
                gt_found = res.get('ground_truth_found', False)
                
                # Format individual CER/WER value
                def format_err_rate_value(value):
                    if not HAS_EDITDISTANCE: return "N/A"
                    if not gt_found: return "NoGT"
                    if isinstance(value, float): return f"{value:.3f}"
                    if value is None: return "CalcErr" # Error during calculation
                    return "" # Should not happen if ground truth was found
                
                # --- Populate row_data with separate CER/WER values ---
                row_data = [
                    idx,
                    img_name[:img_col_width],
                    format_err_rate_value(res.get('question_CER')),
                    format_err_rate_value(res.get('question_WER')),
                    format_err_rate_value(res.get('answer_A_CER')),
                    format_err_rate_value(res.get('answer_A_WER')),
                    format_err_rate_value(res.get('answer_B_CER')),
                    format_err_rate_value(res.get('answer_B_WER')),
                    format_err_rate_value(res.get('answer_C_CER')),
                    format_err_rate_value(res.get('answer_C_WER')),
                    format_err_rate_value(res.get('answer_D_CER')),
                    format_err_rate_value(res.get('answer_D_WER')),
                    color(err_msg, Fore.RED) if err_msg else ('-' if not gt_found else '') 
                ]
                # --- End Row Data Population ---
                table.add_row(row_data)
            print(table)
        else:
            # Fallback: simple text table
            print(header_format_string.format(*headers))
            print("-" * separator_length)
            for idx, res in enumerate(results, 1):
                img_name = res.get('image_filename', '')[:img_col_width]
                err_msg = res.get('error_message', '')
                gt_found = res.get('ground_truth_found', False)
                
                # Format individual CER/WER value (same function as above)
                def format_err_rate_value(value):
                    if not HAS_EDITDISTANCE: return "N/A"
                    if not gt_found: return "NoGT"
                    if isinstance(value, float): return f"{value:.3f}"
                    if value is None: return "CalcErr"
                    return ""

                error_display = color(err_msg, Fore.RED) if err_msg else ('-' if not gt_found else '')

                # --- Populate row_data for manual table ---
                row_data = [
                    idx, img_name,
                    format_err_rate_value(res.get('question_CER')),
                    format_err_rate_value(res.get('question_WER')),
                    format_err_rate_value(res.get('answer_A_CER')),
                    format_err_rate_value(res.get('answer_A_WER')),
                    format_err_rate_value(res.get('answer_B_CER')),
                    format_err_rate_value(res.get('answer_B_WER')),
                    format_err_rate_value(res.get('answer_C_CER')),
                    format_err_rate_value(res.get('answer_C_WER')),
                    format_err_rate_value(res.get('answer_D_CER')),
                    format_err_rate_value(res.get('answer_D_WER')),
                    error_display
                ]
                # --- End Row Data Population ---

                # Ensure enough data points for formatting
                while len(row_data) < len(headers):
                    row_data.append('')
                print(row_format_string.format(*row_data))

        print("\n=============================================================\n")

        # --- Save Detailed Results to CSV --- 
        REPORTS_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = REPORTS_DIR / f"comparison_report_{timestamp}.csv"

        if results:
            # Define explicit fieldnames for CSV clarity
            fieldnames = [
                'image_filename', 'ground_truth_found', 'error_message',
                'gt_question', 'ocr_question', 'question_CER', 'question_WER',
                'gt_answer_A', 'ocr_answer_A', 'answer_A_CER', 'answer_A_WER',
                'gt_answer_B', 'ocr_answer_B', 'answer_B_CER', 'answer_B_WER',
                'gt_answer_C', 'ocr_answer_C', 'answer_C_CER', 'answer_C_WER',
                'gt_answer_D', 'ocr_answer_D', 'answer_D_CER', 'answer_D_WER',
            ]
            
            # Prepare data for CSV writer
            csv_data = []
            for res_dict in results:
                 if isinstance(res_dict, dict):
                     # Select only the desired fields, provide default None if missing
                     row = {key: res_dict.get(key) for key in fieldnames}
                     csv_data.append(row)
                 else:
                     logger.error(f"Found non-dictionary item in results: {res_dict}")
                     csv_data.append({key: 'ERROR_INVALID_DATA' for key in fieldnames})

            if csv_data:
                 try:
                    import csv # Import csv module here
                    with open(report_filename, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(csv_data)
                    logger.info(f"Detailed comparison report saved to: {report_filename}")
                    print(f"Detailed comparison report saved to: {report_filename}")
                 except Exception as e_csv:
                     logger.error(f"Failed to write CSV report: {e_csv}", exc_info=True)
                     print(f"Error: Failed to save detailed report to {report_filename}. Check logs.")
            else:
                 logger.warning("No valid results data to save in the detailed report.")
                 print("Warning: No valid data to save in detailed report.")
        else:
            logger.info("Results list is empty, skipping detailed report saving.")
            print("No results generated, skipping detailed report.")

    except Exception as e:
        logger.error(f"An error occurred during comparison evaluation: {e}", exc_info=True)
        print(f"{Fore.RED}An unexpected error occurred during evaluation: {e}{Style.RESET_ALL}")
# --- Accuracy Evaluation & Reporting end ---


# --- Main Evaluation Logic (Modified) ---
def run_evaluation_on_captures():
    """
    Main function to run the evaluation process on images in the all_captures directory,
    comparing OCR results against the ground_truth.json file.
    """
    logger.info("Starting comparison run...")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Looking for images in: {ALL_CAPTURES_DIR}")
    print(f"Loading config from: {CONFIG_FILE}")
    print(f"Loading Ground Truth from: {GROUND_TRUTH_FILE}") # Updated message

    # --- 1. Load Configuration --- 
    config = None
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            logger.info("Configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load config file {CONFIG_FILE}: {e}", exc_info=True)
            print(f"{Fore.RED}Error loading config: {e}{Style.RESET_ALL}")
            return # Cannot proceed without config for regions
    else:
        logger.error(f"Config file not found: {CONFIG_FILE}")
        print(f"{Fore.RED}Error: Configuration file '{CONFIG_FILE.name}' not found in project root.{Style.RESET_ALL}")
        return

    # Validate essential config keys (still needed for cropping)
    if not all(k in config for k in ['question_region', 'answer_regions']):
         logger.error("Config file missing 'question_region' or 'answer_regions'.")
         print(f"{Fore.RED}Error: Config file must contain 'question_region' and 'answer_regions'.{Style.RESET_ALL}")
         return
    if not isinstance(config.get('answer_regions'), dict) or not config['answer_regions']:
        logger.error("'answer_regions' in config is not a valid dictionary.")
        print(f"{Fore.RED}Error: 'answer_regions' in config must be a dictionary with entries like 'A': {{...}}, 'B': {{...}}.{Style.RESET_ALL}")
        return

    # --- 2. Load Ground Truth JSON --- 
    ground_truth_map = {}
    if GROUND_TRUTH_FILE.exists():
        try:
            with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
                ground_truth_list = json.load(f)
                if isinstance(ground_truth_list, list):
                    # Convert list to dict keyed by image_file for fast lookup
                    for item in ground_truth_list:
                        if isinstance(item, dict) and 'image_file' in item:
                            ground_truth_map[item['image_file']] = item
                    logger.info(f"Ground truth loaded successfully with {len(ground_truth_map)} entries.")
                    if not ground_truth_map:
                        logger.warning("Ground truth file loaded but is empty or contains no valid entries.")
                        print(f"{Fore.YELLOW}Warning: Ground truth file '{GROUND_TRUTH_FILE.name}' is empty or invalid.{Style.RESET_ALL}")
                else:
                    logger.error(f"Ground truth file '{GROUND_TRUTH_FILE.name}' is not a JSON list.")
                    print(f"{Fore.RED}Error: Ground truth file '{GROUND_TRUTH_FILE.name}' must contain a JSON list of objects.{Style.RESET_ALL}")
                    # Continue without ground truth? Or exit? For evaluation, maybe exit.
                    return
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode ground truth JSON file {GROUND_TRUTH_FILE}: {e}", exc_info=True)
             print(f"{Fore.RED}Error decoding ground truth JSON: {e}{Style.RESET_ALL}")
             return
        except Exception as e:
            logger.error(f"Failed to load ground truth file {GROUND_TRUTH_FILE}: {e}", exc_info=True)
            print(f"{Fore.RED}Error loading ground truth file: {e}{Style.RESET_ALL}")
            return
    else:
        logger.error(f"Ground truth file not found: {GROUND_TRUTH_FILE}")
        print(f"{Fore.RED}Error: Ground truth file '{GROUND_TRUTH_FILE.name}' not found.{Style.RESET_ALL}")
        return # Cannot proceed without ground truth for comparison

    # --- 3. Initialize OCR Processor --- 
    try:
        ocr_processor = OCRProcessor()
        logger.info("OCR Processor initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize OCR Processor: {e}", exc_info=True)
        print(f"{Fore.RED}Fatal Error: Could not initialize OCR Processor. Check logs. Exiting.{Style.RESET_ALL}")
        sys.exit(1)

    # --- 4. TF-IDF Removed --- 

    # --- 5. Find Images --- 
    if not ALL_CAPTURES_DIR.exists():
        logger.error(f"Image directory not found: {ALL_CAPTURES_DIR}")
        print(f"{Fore.RED}Error: Directory '{ALL_CAPTURES_DIR.name}' not found.{Style.RESET_ALL}")
        return

    image_files = list(ALL_CAPTURES_DIR.glob('*.png')) + list(ALL_CAPTURES_DIR.glob('*.jpg')) + list(ALL_CAPTURES_DIR.glob('*.jpeg'))
    if not image_files:
        logger.warning(f"No image files found in {ALL_CAPTURES_DIR}")
        print(f"{Fore.YELLOW}Warning: No images (.png, .jpg, .jpeg) found in '{ALL_CAPTURES_DIR.name}'.{Style.RESET_ALL}")
        return

    logger.info(f"Found {len(image_files)} images to process.")
    print(f"Found {len(image_files)} images in '{ALL_CAPTURES_DIR.name}'. Starting processing...")

    # --- 6. Process Each Image --- 
    all_results = []
    processed_count = 0
    error_count = 0

    for image_path in image_files:
        image_filename = image_path.name
        logger.info(f"Processing image: {image_filename}")
        print(f"\n--- Processing: {image_filename} ---")
        
        # Initialize result data structure for this image
        result_data = {
            'image_filename': image_filename,
            'ground_truth_found': False,
            'error_message': None,
            'gt_question': None,
            'ocr_question': None,
            'gt_answer_A': None,
            'ocr_answer_A': None,
            # ... Add placeholders for B, C, D ground truth and OCR 
            'gt_answer_B': None,
            'ocr_answer_B': None,
            'gt_answer_C': None,
            'ocr_answer_C': None,
            'gt_answer_D': None,
            'ocr_answer_D': None,
        }

        try:
            # --- Get Ground Truth for this image ---
            ground_truth_entry = ground_truth_map.get(image_filename)
            if ground_truth_entry:
                 result_data['ground_truth_found'] = True
                 result_data['gt_question'] = ground_truth_entry.get('question_text')
                 gt_answers = ground_truth_entry.get('answer_texts', {})
                 for choice in ['A', 'B', 'C', 'D']:
                     result_data[f'gt_answer_{choice}'] = gt_answers.get(choice)
                 logger.info(f"Ground truth found for {image_filename}")
                 print(f"  Ground Truth Found.")
                 # Optional: print GT Question for quick check
                 # print(f"    GT Q: {result_data['gt_question'][:80]}...")
            else:
                 logger.warning(f"No ground truth entry found for {image_filename} in {GROUND_TRUTH_FILE.name}")
                 print(f"  {Fore.YELLOW}Warning: No ground truth found for this image.{Style.RESET_ALL}")
                 # Still process OCR, but comparison won't happen in evaluate_accuracy

            # --- Load and OCR Image Regions ---
            full_image = cv2.imread(str(image_path))
            if full_image is None:
                raise ValueError("Could not load image using OpenCV.")

            regions_to_ocr = {}
            q_region = config['question_region']
            # Check if region dimensions are valid before slicing
            if q_region['y'] + q_region['height'] <= full_image.shape[0] and q_region['x'] + q_region['width'] <= full_image.shape[1]:
                 regions_to_ocr['question'] = full_image[q_region['y']:q_region['y']+q_region['height'], q_region['x']:q_region['x']+q_region['width']]
            else:
                 logger.error(f"Question region coordinates are out of bounds for image {image_filename}. Skipping region.")
                 regions_to_ocr['question'] = None # Indicate missing region

            for ans_label, ans_region in config['answer_regions'].items():
                 if ans_region['y'] + ans_region['height'] <= full_image.shape[0] and ans_region['x'] + ans_region['width'] <= full_image.shape[1]:
                     regions_to_ocr[ans_label] = full_image[ans_region['y']:ans_region['y']+ans_region['height'], ans_region['x']:ans_region['x']+ans_region['width']]
                 else:
                     logger.error(f"Answer region '{ans_label}' coordinates are out of bounds for image {image_filename}. Skipping region.")
                     regions_to_ocr[ans_label] = None # Indicate missing region

            # Perform OCR - Handle potential None regions gracefully if needed by OCRProcessor
            ocr_texts = ocr_processor.process_quiz_regions(regions_to_ocr)

            # Apply filter_selected_pattern if enabled in config
            should_filter_selected = config.get('filter_selected_pattern', True)
            
            # Store OCR results, applying filter if needed
            raw_question_text = ocr_texts.get('question', '')
            result_data['ocr_question'] = filter_selected_pattern(raw_question_text) if should_filter_selected else raw_question_text
            
            for choice in ['A', 'B', 'C', 'D']:
                raw_answer_text = ocr_texts.get(choice, '')
                result_data[f'ocr_answer_{choice}'] = filter_selected_pattern(raw_answer_text) if should_filter_selected else raw_answer_text

            logger.debug(f"OCR Question: {result_data['ocr_question']}")
            print(f"  OCR Question: {result_data['ocr_question'][:80]}{'...' if len(result_data['ocr_question']) > 80 else ''}")
            if should_filter_selected:
                print(f"  Applied '[number] selected' pattern filtering.")
            # Optional: Print OCR answers
            # for choice in ['A', 'B', 'C', 'D']:
            #     print(f"    OCR Ans {choice}: {result_data[f'ocr_answer_{choice}']}")

            # --- Comparison Logic Removed (Moved to evaluate_accuracy) ---

        except Exception as e:
            logger.error(f"Error processing image {image_filename}: {e}", exc_info=True)
            result_data['error_message'] = str(e)
            error_count += 1
            print(f"  {Fore.RED}Error processing image: {e}{Style.RESET_ALL}")

        finally:
            all_results.append(result_data)
            processed_count += 1
            # Optional: Clear GPU memory periodically
            # if processed_count % 10 == 0:
            #     # Add memory clearing logic if needed (e.g., torch.cuda.empty_cache())
            #     pass 

    logger.info(f"Finished processing {processed_count} images with {error_count} errors.")
    print(f"\nFinished processing {processed_count} images ({error_count} errors encountered).")

    # --- 7. Evaluate and Report --- 
    if all_results:
        evaluate_accuracy(all_results, PROJECT_ROOT)
    else:
        logger.warning("No results were generated from image processing.")
        print("No results generated, skipping evaluation report.")

# --- Main Execution --- 
if __name__ == "__main__":
    run_evaluation_on_captures()
# --- Main Execution end ---
