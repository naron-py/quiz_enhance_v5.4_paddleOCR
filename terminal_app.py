import cv2
import numpy as np
import pandas as pd
import json
import os
from PIL import Image, ImageTk
import tkinter as tk
import pyautogui
import mss
pyautogui.PAUSE = 0
pyautogui.MINIMUM_DURATION = 0
import difflib
import re
from ocr_processor import OCRProcessor
from config_manager import ConfigManager
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading
from pynput import keyboard
import logging
import sys
import gc  # For garbage collection
from datetime import datetime # Added for timestamped filenames
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Import Rich for colored terminal output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.box import Box, ROUNDED, SIMPLE
console = Console()

# Configuration file path
CONFIG_FILE = 'config.json'
config_manager = ConfigManager(CONFIG_FILE)
config = config_manager.data

# Set up logging based on configuration
log_level_name = config.get('log_level', 'INFO').upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('terminal_app.log'),
        logging.StreamHandler()
    ]
)

# Try to import torch for GPU memory management
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Try to import Windows-specific modules for direct input
try:
    import win32api, win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

# Global variables
ocr_processor = None
questions_df = None
captured_images = {}
recognized_text = {}
best_match = None
tfidf_vectorizer = None
tfidf_matrix = None
timings = {}
auto_click = False  # Whether to automatically click on the answer
question_capture_count = 0 # New global counter for captured questions
active_database = "default"  # Tracks which database is currently loaded
spam_capture_mode = False  # Flag for controlling continuous spam capture mode
spam_capture_thread = None  # Thread for running spam capture

# For thread safety
capture_lock = threading.Lock()

# --- Debug Directory ---
debug_dir = Path('debug_images')
debug_dir.mkdir(exist_ok=True)
# --- Debug Directory end ---

def capture_screen(region):
    """Capture a specific region of the screen"""
    with mss.mss() as sct:
        bbox = {
            "left": region['x'],
            "top": region['y'],
            "width": region['width'],
            "height": region['height']
        }
        screenshot = sct.grab(bbox)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)

def click_on_answer(region):
    """Click in the middle of the specified answer region"""
    try:
        # Calculate the center of the region
        center_x = region['x'] + region['width'] // 2
        center_y = region['y'] + region['height'] // 2
        
        # Log the click attempt
        logging.info(f"Attempting to click at position: ({center_x}, {center_y})")
        
        # Try different click methods based on available libraries
        click_success = False
        
        # Method 1: Windows API direct click (most reliable for games)
        if HAS_WIN32:
            try:
                # Move to target position and click directly (no position storage for speed)
                win32api.SetCursorPos((center_x, center_y))
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
                # Minimal delay between down and up for reliability
                time.sleep(0.01)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
                
                # Log success
                logging.info("Click performed using win32api")
                click_success = True
            except Exception as e:
                logging.error(f"Win32 click failed: {e}")
        
        # Method 2: PyAutoGUI fallback
        if not click_success:
            try:
                pyautogui.click(x=center_x, y=center_y)
                time.sleep(0.01)  # Minimal delay for reliability

                # Log success
                logging.info("Click performed using pyautogui")
                click_success = True
            except Exception as e:
                logging.error(f"PyAutoGUI click failed: {e}")
        
        # Return success status
        return click_success
    except Exception as e:
        logging.error(f"Error in click_on_answer: {e}")
        return False

def load_questions_data():
    """Load questions and answers from CSV based on active database setting"""
    global active_database, config
    
    # Determine which database to load based on configuration
    db_setting = config.get('active_database', 'default').lower()
    active_database = db_setting
    
    try:
        if db_setting == 'magic':
            # Load magic database
            df = pd.read_csv('HPMA_data_magic.csv', sep='$', names=['question', 'answer'])
            logging.info(f"Loaded magic database with {len(df)} questions")
            return df
        elif db_setting == 'muggle':
            # Load muggle database
            df = pd.read_csv('HPMA_data_muggle.csv', sep='$', names=['question', 'answer'])
            logging.info(f"Loaded muggle database with {len(df)} questions")
            return df
        elif db_setting == 'all':
            # Load both databases and combine them
            magic_df = pd.read_csv('HPMA_data_magic.csv', sep='$', names=['question', 'answer'])
            muggle_df = pd.read_csv('HPMA_data_muggle.csv', sep='$', names=['question', 'answer'])
            df = pd.concat([magic_df, muggle_df], ignore_index=True)
            logging.info(f"Loaded combined database with {len(df)} questions ({len(magic_df)} magic + {len(muggle_df)} muggle)")
            return df
        else:
            # Default case - try to load combined file first, fall back to creating it
            try:
                df = pd.read_csv('HPMA_data.csv', sep='$', names=['question', 'answer'])
                logging.info(f"Loaded default database with {len(df)} questions")
                return df
            except FileNotFoundError:
                # Create combined file if it doesn't exist
                try:
                    magic_df = pd.read_csv('HPMA_data_magic.csv', sep='$', names=['question', 'answer'])
                    muggle_df = pd.read_csv('HPMA_data_muggle.csv', sep='$', names=['question', 'answer'])
                    df = pd.concat([magic_df, muggle_df], ignore_index=True)
                    # Save the combined file for future use
                    df.to_csv('HPMA_data.csv', sep='$', index=False, header=False)
                    logging.info(f"Created and loaded combined database with {len(df)} questions")
                    return df
                except Exception as e:
                    logging.error(f"Error creating combined database: {str(e)}")
                    return None
    except FileNotFoundError as e:
        logging.error(f"Database file not found: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error loading database: {str(e)}")
        return None

def switch_database(db_name):
    """Switch to a different database and reload"""
    global config, questions_df, tfidf_vectorizer, tfidf_matrix, active_database
    
    if db_name.lower() not in ['default', 'magic', 'muggle', 'all']:
        console.print(f"[bold red]Error:[/bold red] Invalid database name. Use 'default', 'magic', 'muggle', or 'all'")
        return False
    
    # Update config
    config['active_database'] = db_name.lower()
    active_database = db_name.lower()
    config_manager.data = config
    config_manager.save()
    
    # Reload database and compute TF-IDF
    console.print(f"[cyan]Switching to {db_name} database...[/cyan]")
    questions_df = load_questions_data()

    if questions_df is not None:
        console.print(f"[green]Loaded {len(questions_df)} questions from {db_name} database.[/green]")
        
        # Recompute TF-IDF Matrix
        try:
            tfidf_vectorizer, tfidf_matrix = compute_tfidf_matrix(questions_df)
            logging.info("TF-IDF matrix computed successfully.")
            return True
        except Exception as e:
            console.print(f"[bold yellow]Warning:[/bold yellow] Failed to compute TF-IDF matrix: {e}. Matching may be less reliable.")
            logging.error(f"Failed to compute TF-IDF matrix: {e}", exc_info=True)
            tfidf_vectorizer, tfidf_matrix = None, None
            return False
    else:
        console.print(f"[bold red]Error:[/bold red] Failed to load {db_name} database.")
        return False

def compute_tfidf_matrix(questions_df):
    """Compute TF-IDF matrix for all questions"""
    vectorizer = TfidfVectorizer().fit(questions_df['question'].astype(str))
    tfidf_matrix = vectorizer.transform(questions_df['question'].astype(str))
    return vectorizer, tfidf_matrix

def find_best_match_tfidf(text, questions_df, vectorizer, tfidf_matrix):
    """Find best match using TF-IDF + cosine similarity"""
    if questions_df is None or text is None:
        return None, None, 0
    try:
        query_vec = vectorizer.transform([text])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        if best_score >= 0.85:  # Lower threshold for TF-IDF
            row = questions_df.iloc[best_idx]
            return row['question'], row['answer'], best_score
    except Exception as e:
        logging.error(f"Error in TF-IDF matching: {str(e)}")
    return None, None, 0

def find_all_matching_questions(text, questions_df, vectorizer, tfidf_matrix, threshold=0.85):
    """Find all questions that match the given text with similarity above threshold"""
    matching_entries = []
    if questions_df is None or text is None or len(text) < 5:
        return matching_entries
    
    try:
        query_vec = vectorizer.transform([text])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Find all indices where similarity is above threshold
        matching_indices = np.where(similarities >= threshold)[0]
        
        # Sort by similarity score (highest first)
        matching_indices = sorted(matching_indices, key=lambda i: similarities[i], reverse=True)
        
        # Return all matching entries
        for idx in matching_indices:
            row = questions_df.iloc[idx]
            matching_entries.append({
                'question': row['question'],
                'answer': row['answer'],
                'score': similarities[idx]
            })
            
        return matching_entries
    except Exception as e:
        logging.error(f"Error finding all matching questions: {str(e)}")
        return []

def get_text_similarity(text1, text2):
    """Get similarity between two texts using multiple methods"""
    if not text1 or not text2:
        return 0
    
    text1 = text1.lower()
    text2 = text2.lower()
    
    # Exact match
    if text1 == text2:
        return 1.0
    
    # Sequence matcher similarity
    seq_similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
    
    # Word-based similarity
    words1 = set(text1.split())
    words2 = set(text2.split())
    word_similarity = len(words1.intersection(words2)) / max(len(words1), len(words2))
    
    # Token-based similarity (handling partial words)
    tokens1 = set(''.join(c for c in text1 if c.isalnum()).lower())
    tokens2 = set(''.join(c for c in text2 if c.isalnum()).lower())
    token_similarity = len(tokens1.intersection(tokens2)) / max(len(tokens1), len(tokens2))
    
    # Return weighted average of similarities
    return (seq_similarity * 0.6 + word_similarity * 0.2 + token_similarity * 0.2)

def find_best_match(text, q_df):
    """Find best match using TF-IDF and then refine with similarity"""
    start_time = time.time()
    global best_match

    if text is None or not text.strip():
        logging.warning("Attempted to find match for empty or None text.")
        return None, None, 0
    
    # Basic text cleaning before matching
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    if not cleaned_text.strip():
        logging.warning(f"Text '{text}' became empty after cleaning.")
        return None, None, 0

    # Use our improved matching function
    matching_entries = find_all_matching_questions(cleaned_text, q_df, tfidf_vectorizer, tfidf_matrix)
    
    if matching_entries:
        # Take the highest scoring match
        match_entry = matching_entries[0]
        match_q = match_entry['question']
        match_a = match_entry['answer']
        score_tfidf = match_entry['score']

        # Store the best match found
        best_match = {
            'ocr_question': text, 
            'db_question': match_q, 
            'db_answer': match_a, 
            'score': score_tfidf,
            'all_matches': matching_entries[:3]  # Keep top 3 for reference
        }
    else:
        match_q, match_a, score_tfidf = None, None, 0
        best_match = {
            'ocr_question': text, 
            'db_question': None, 
            'db_answer': None, 
            'score': 0,
            'all_matches': []
        }

    end_time = time.time()
    timings['find_best_match'] = end_time - start_time

    # Return the top result
    return match_q, match_a, score_tfidf

def clear_gpu_memory():
    """Clear GPU memory if torch is available"""
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("Cleared GPU memory.")

def capture_and_save_fullscreen_on_nomatch():
    """Captures the entire screen and saves it when no DB match is found."""
    global question_capture_count
    try:
        # Ensure the directory exists
        save_dir = Path('not_found_pic')
        save_dir.mkdir(exist_ok=True)

        # Increment counter and generate filename
        question_capture_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = save_dir / f"{question_capture_count:03d}_fullscreen_capture_{timestamp}.png"

        # Capture fullscreen
        fullscreen_img = pyautogui.screenshot()
        fullscreen_img_np = cv2.cvtColor(np.array(fullscreen_img), cv2.COLOR_RGB2BGR)

        # Save the image
        cv2.imwrite(str(filename), fullscreen_img_np)
        logging.info(f"Saved full screen capture due to no match: {filename}")
        console.print(f"[yellow]No match found. Saved fullscreen capture: {filename}[/yellow]")
    except Exception as e:
        logging.error(f"Failed to capture/save fullscreen on no match: {e}", exc_info=True)
        console.print(f"[bold red]Error saving fullscreen capture:[/bold red] {e}")

def save_processed_images(capture_count: int):
    """Saves fullscreen image to all_captures directory."""
    try:
        save_dir = Path('all_captures')
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Use capture_count for ordering
        base_filename = f"{capture_count:03d}_capture_{timestamp}"

        # Save only the fullscreen capture
        fullscreen_img = pyautogui.screenshot()
        fullscreen_img_np = cv2.cvtColor(np.array(fullscreen_img), cv2.COLOR_RGB2BGR)
        fullscreen_filename = save_dir / f"{base_filename}_fullscreen.png"
        cv2.imwrite(str(fullscreen_filename), fullscreen_img_np)
        
        console.print(f"[cyan]Saved fullscreen image to: {fullscreen_filename}[/cyan]")
        logging.info(f"Saved fullscreen image for sequence {capture_count}: {fullscreen_filename}")

    except Exception as e:
        logging.error(f"Failed to save fullscreen image: {e}", exc_info=True)
        console.print(f"[bold red]Error saving fullscreen image:[/bold red] {e}")

def crop_image_from_numpy(full_image_np: np.ndarray, region: dict) -> Optional[np.ndarray]:
    """Crops a region from a numpy array representing an image."""
    try:
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        # Ensure coordinates are within bounds
        img_h, img_w = full_image_np.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        if x1 >= x2 or y1 >= y2:
            logging.warning(f"Invalid crop dimensions for region {region}. Resulting crop is empty.")
            return None
            
        return full_image_np[y1:y2, x1:x2]
    except Exception as e:
        logging.error(f"Error cropping image from numpy array for region {region}: {e}", exc_info=True)
        return None

def filter_selected_pattern(text):
    """Filter out '[number] selected' pattern from text"""
    if not text:
        return text
    
    # Pattern: matches text ending with 1 or more digits followed by "selected" (with optional space)
    pattern = r'\s*\d+\s*selected\s*$'
    return re.sub(pattern, '', text).strip()

def capture_and_process():
    """Main function to capture, process, and find match"""
    start_total_time = time.time()
    global captured_images, recognized_text, best_match, question_capture_count
    timings.clear() # Reset timings for this cycle
    
    with capture_lock:
        # 1. Capture images
        capture_start = time.time()
        captured_images.clear() # Clear previous captures
        try:
            captured_images['question'] = capture_screen(config['question_region'])
            for label, region in config['answer_regions'].items():
                captured_images[label] = capture_screen(region)
            capture_end = time.time()
            timings['capture'] = capture_end - capture_start
            logging.info("Screen regions captured.")
        except Exception as e:
            logging.error(f"Error during screen capture: {e}", exc_info=True)
            console.print(f"[bold red]Error during screen capture:[/bold red] {e}")
            return

        # 2. OCR Processing
        ocr_start = time.time()
        recognized_text.clear()
        try:
            # Prepare regions for batch processing
            regions_to_ocr = {}
            if captured_images.get('question') is not None:
                 regions_to_ocr['question'] = captured_images['question']
            for label in ['A', 'B', 'C', 'D']:
                 if captured_images.get(label) is not None:
                     regions_to_ocr[label] = captured_images[label]
            
            # Perform OCR using the batch method
            ocr_texts = ocr_processor.process_quiz_regions(regions_to_ocr)
            
            # Store results
            recognized_text['question'] = ocr_texts.get('question', '')
            for label in ['A', 'B', 'C', 'D']:
                 recognized_text[label] = ocr_texts.get(label, '')
                 
                 # Optional: Filter out A./B./etc. tags if enabled
                 if config.get('filter_answer_choice_tags', True):
                      # Simple filter: remove first 2 chars if they match pattern like "A."
                      if len(recognized_text[label]) > 2 and recognized_text[label][0].isalpha() and recognized_text[label][1] in '. ':
                          recognized_text[label] = recognized_text[label][2:].strip()
                 
                 # Filter out [number] selected pattern if enabled
                 if config.get('filter_selected_pattern', True):
                     recognized_text[label] = filter_selected_pattern(recognized_text[label])
                          
            ocr_end = time.time()
            timings['ocr'] = ocr_end - ocr_start
            logging.info(f"OCR processing complete. Question: {recognized_text.get('question', 'N/A')}")
            
            # Create and display a table with OCR results
            console.print(f"[cyan][{question_capture_count}] Question:[/cyan] {recognized_text.get('question', '[OCR Failed]')}")
            
            if config.get('show_ocr_answer_choices_terminal', True):
                # Create a Rich table for options
                table = Table(show_header=True, box=ROUNDED, border_style="cyan")
                table.add_column("Choice", style="bold cyan", justify="center")
                table.add_column("Text", style="white")
                
                for label in ['A', 'B', 'C', 'D']:
                    table.add_row(label, recognized_text.get(label, '[OCR Failed]'))
                
                console.print(table)
                    
        except Exception as e:
            logging.error(f"Error during OCR processing: {e}", exc_info=True)
            console.print(f"[bold red]Error during OCR processing:[/bold red] {e}")
            return
            
        # Increment capture count AFTER successful capture and OCR
        question_capture_count += 1

        # Save images if configured
        if config.get('save_all_captured_images', False):
            save_processed_images(question_capture_count) # Pass the current count

        # 3. Find Best Match
        match_start = time.time()
        ocr_question_text = recognized_text.get('question')
        if ocr_question_text and questions_df is not None and tfidf_vectorizer is not None:
            # Using the new function to find all matching questions
            matching_entries = find_all_matching_questions(ocr_question_text, questions_df, tfidf_vectorizer, tfidf_matrix)
            match_end = time.time()
            timings['matching'] = match_end - match_start
            
            if matching_entries:
                # First try exact match with answers
                best_match_entry = None
                best_match_choice = None
                best_match_similarity = -1.0
                
                for entry in matching_entries:
                    match_q = entry['question']
                    match_a = entry['answer']
                    score = entry['score']
                    
                    # Try to find matching answer choice for this potential answer
                    for label in ['A', 'B', 'C', 'D']:
                        ocr_answer_text = recognized_text.get(label)
                        if ocr_answer_text:
                            similarity = get_text_similarity(match_a, ocr_answer_text)
                            if similarity > best_match_similarity:
                                best_match_similarity = similarity
                                best_match_choice = label
                                best_match_entry = entry
                
                # Use a reasonable threshold
                ANSWER_SIMILARITY_THRESHOLD = 0.7
                
                if best_match_entry and best_match_choice and best_match_similarity >= ANSWER_SIMILARITY_THRESHOLD:
                    # Found a good match both for question and answer
                    match_q = best_match_entry['question']
                    match_a = best_match_entry['answer']
                    score = best_match_entry['score']
                    
                    logging.info(f"Match found: DB Q='{match_q}', DB A='{match_a}', Score={score:.3f}, Choice={best_match_choice}, Similarity={best_match_similarity:.3f}")
                    
                    # Create a match result panel
                    match_panel = Panel(
                        f"[bold white]Database Answer:[/bold white] [bold green]{match_a}[/bold green]\n"
                        f"[bold white]Question Match:[/bold white] {match_q}\n"
                        f"[bold white]Best Choice:[/bold white] [bold cyan]{best_match_choice}[/bold cyan] (Similarity: {best_match_similarity:.3f})",
                        title="[bold green]Match Found[/bold green]",
                        border_style="green"
                    )
                    console.print(match_panel)
                    
                    # Auto-click if enabled
                    if auto_click:
                        click_start = time.time()
                        if click_on_answer(config['answer_regions'][best_match_choice]):
                            console.print(f"[green]Auto-clicked on region {best_match_choice}.[/green]")
                            logging.info(f"Auto-clicked on region {best_match_choice}.")
                        else:
                            console.print(f"[bold red]Auto-click failed for region:[/bold red] {best_match_choice}")
                            logging.warning(f"Auto-click failed for region {best_match_choice}")
                        timings['auto_click'] = time.time() - click_start
                else:
                    # Question matched but couldn't find a matching answer choice for any potential answer
                    logging.warning(f"Found {len(matching_entries)} question matches but could not reliably identify any matching answer choice.")
                    
                    # For debugging, show the best we found even if below threshold
                    debug_info = ""
                    if best_match_entry:
                        debug_info = f"Best was '{best_match_entry['answer']}' matching to choice {best_match_choice} with similarity {best_match_similarity:.3f}"
                    
                    # Create a table for possible answers
                    possible_answers = Table(show_header=True, box=ROUNDED, title="[bold red]Possible Database Answers[/bold red]", border_style="red")
                    possible_answers.add_column("#", style="dim", justify="right")
                    possible_answers.add_column("Answer", style="bold green")
                    possible_answers.add_column("Score", justify="right")
                    
                    for idx, entry in enumerate(matching_entries[:5], 1):  # Show up to 5 possible answers
                        possible_answers.add_row(str(idx), entry['answer'], f"{entry['score']:.3f}")
                    
                    # Error panel
                    error_panel = Panel(
                        f"Question: '{matching_entries[0]['question']}'\n"
                        f"Highest Choice Similarity: [bold red]{best_match_similarity:.3f}[/bold red] for {best_match_choice} (threshold: {ANSWER_SIMILARITY_THRESHOLD})",
                        title="[bold red]❌ Matching Question But No Matching Choice Found[/bold red]",
                        border_style="red"
                    )
                    
                    console.print(error_panel)
                    console.print(possible_answers)
                    
                    # Save fullscreen capture for question matched but no choice matched
                    if config.get('capture_fullscreen_on_nomatch', False):
                        capture_and_save_fullscreen_on_nomatch()
            else: # No matching questions found
                logging.warning(f"No database match found for OCR question: '{ocr_question_text}'")
                
                # Error panel for no match
                no_match_panel = Panel(
                    "[red]Database Answer (from initial question match): [italic]None[/italic][/red]",
                    title="[bold red]❌ No Matching Answer Found[/bold red]",
                    border_style="red"
                )
                console.print(no_match_panel)
                
                # Optional: Trigger fullscreen capture if enabled
                if config.get('capture_fullscreen_on_nomatch', False):
                     capture_and_save_fullscreen_on_nomatch()
        else:
            match_end = time.time() # Still record time even if no match attempted
            timings['matching'] = match_end - match_start
            logging.warning("Skipping match finding: OCR question text is empty or database/TF-IDF not loaded.")
            # Wrap console print in try-except to handle potential NoneType during shutdown
            console.print("[yellow]Skipping match finding (Empty OCR / No DB?).[/yellow]")

    # Clear GPU memory periodically
    if question_capture_count % 5 == 0: # Adjust frequency as needed
        clear_gpu_memory()
        
    # Print timings if enabled
    end_total_time = time.time()
    timings['total_cycle'] = end_total_time - start_total_time
    if config.get('show_processing_times', True):
        timing_panel = Panel(
            " | ".join([f"{k}: {v:.3f}s" for k, v in timings.items()]),
            title="[bold blue]Processing Times[/bold blue]",
            border_style="blue"
        )
        console.print(timing_panel)

def on_press(key):
    """Handle key presses"""
    try:
        if key == keyboard.Key.f2: # F2 captures and processes
            print("\n--- F2 Pressed: Capturing and Processing ---")
            capture_and_process()
        elif key == keyboard.Key.f3: # F3 now reloads instead of F4
            print("\n--- F3 Pressed: Reloading Configuration ---")
            initialize()
        elif key == keyboard.Key.f10: # F10 toggles spam capture mode
            print("\n--- F10 Pressed: Toggling Spam Capture Mode ---")
            toggle_spam_capture_mode()
    except AttributeError:
        # Handle regular keys if needed, e.g., key.char
        pass
    except Exception as e:
        logging.error(f"Error in key press handler: {e}", exc_info=True)
        
    # Continue listening
    return True



def spam_capture_loop():
    """Continuously capture and process quiz questions without any delay between captures"""
    global spam_capture_mode
    
    console.print("[bold green]Starting spam capture mode[/bold green]")
    console.print("[bold cyan]Press F10 to stop spam capture mode[/bold cyan]")
    
    while spam_capture_mode:
        try:
            # Process one capture cycle
            result = capture_and_process()
            
            # No delay between captures - immediately start next capture
            # (removed the time.sleep() to fulfill requirement for continuous captures)
            
        except Exception as e:
            logging.error(f"Error in spam capture loop: {e}", exc_info=True)
            console.print(f"[bold red]Error in spam capture:[/bold red] {e}")
            time.sleep(1)  # Pause briefly on error to prevent error flood
    
    console.print("[bold yellow]Spam capture mode stopped[/bold yellow]")



def toggle_spam_capture_mode():
    """Toggle the spam capture mode on/off"""
    global spam_capture_mode, spam_capture_thread
    
    # Toggle the mode
    spam_capture_mode = not spam_capture_mode
    
    if spam_capture_mode:
        # Start the spam capture thread if it doesn't exist or is not alive
        if spam_capture_thread is None or not spam_capture_thread.is_alive():
            spam_capture_thread = threading.Thread(target=spam_capture_loop, daemon=True)
            spam_capture_thread.start()
        console.print("[bold green]Spam capture mode enabled[/bold green]")
    else:
        console.print("[bold yellow]Spam capture mode disabled[/bold yellow]")
        
    return spam_capture_mode



def initialize():
    """Initialize the application by loading configuration and data"""
    global config, ocr_processor, questions_df, tfidf_vectorizer, tfidf_matrix, auto_click

    # First, load the configuration
    config_manager.load()
    config = config_manager.data
    
    # Set auto_click based on configuration
    auto_click = config.get('auto_click', False)
    
    # Initialize OCR processor
    try:
        from ocr_processor import OCRProcessor
        global ocr_processor
        ocr_processor = OCRProcessor()
        logging.info("OCR Processor initialized.")
    except Exception as e:
        logging.error(f"Error initializing OCR processor: {e}", exc_info=True)
    
    # Load questions data
    questions_df = load_questions_data()
    
    if questions_df is not None:
        # Compute TF-IDF matrices for matching
        try:
            tfidf_vectorizer, tfidf_matrix = compute_tfidf_matrix(questions_df)
            logging.info(f"TF-IDF matrix computed with {len(questions_df)} questions.")
        except Exception as e:
            logging.error(f"Error computing TF-IDF matrix: {e}", exc_info=True)
    else:
        logging.warning("Failed to load questions data.")
    
    logging.info(f"Initialization complete. Active database: {active_database}")

def show_config():
    """Display current configuration"""
    print("\n--- Current Configuration ---")
    if config:
        # Use rich console for better JSON printing if available
        console.print_json(data=config)
    else:
        print("Configuration not loaded.")
    print("---------------------------")

def set_config(args):
    """Set a configuration value"""
    if len(args) < 2:
        print("Usage: set <key> <value>")
        return

    key = args[0]
    value_str = ' '.join(args[1:])

    if not config:
        print("Configuration not loaded. Cannot set value.")
        return
        
    # Check if the key exists at the top level
    if key not in config:
        # Check if it's a nested key (e.g., question_region.x)
        if '.' in key:
            parts = key.split('.', 1)
            main_key = parts[0]
            nested_key = parts[1]
            if main_key in config and isinstance(config[main_key], dict) and nested_key in config[main_key]:
                try:
                    # Attempt to parse the value as JSON (int, float, bool, string)
                    try:
                        value = json.loads(value_str)
                    except json.JSONDecodeError:
                        value = value_str  # Keep as string if not valid JSON

                    # Set the nested value
                    config[main_key][nested_key] = value
                    config_manager.data = config
                    config_manager.save()
                    console.print(f"[green]Set {key} = {value}[/green]")
                except Exception as e:
                    console.print(f"[bold red]Error setting nested config value:[/bold red] {e}")
            else:
                console.print(f"[bold red]Error: Key '{key}' not found in configuration.[/bold red]")
        else:
            console.print(f"[bold red]Error: Key '{key}' not found in configuration.[/bold red]")
        return

    # Handle top-level key
    try:
        # Attempt to parse the value as JSON (int, float, bool, string)
        try:
            value = json.loads(value_str)
        except json.JSONDecodeError:
            # Special handling for boolean strings if JSON parsing fails
            if value_str.lower() == 'true':
                 value = True
            elif value_str.lower() == 'false':
                 value = False
            else:
                 value = value_str # Keep as string if not valid JSON
        
        # Specific handling for boolean toggles if needed
        if key == 'auto_click' and isinstance(value, bool):
             global auto_click
             auto_click = value
        config[key] = value
        config_manager.data = config
        config_manager.save()
        console.print(f"[green]Set {key} = {value}[/green]")
    except Exception as e:
        console.print(f"[bold red]Error setting config value:[/bold red] {e}")

def toggle_auto_click():
    """Toggle auto click setting"""
    global auto_click, config
    auto_click = not auto_click
    config['auto_click'] = auto_click # Update config dict
    config_manager.data = config
    config_manager.save()
    print(f"Auto click {'enabled' if auto_click else 'disabled'}.")

def toggle_filter_selected():
    """Toggle filter selected pattern setting"""
    global config
    config['filter_selected_pattern'] = not config.get('filter_selected_pattern', True)
    config_manager.data = config
    config_manager.save()
    enabled_status = "enabled" if config['filter_selected_pattern'] else "disabled"
    console.print(f"[bold cyan]Filter '[number] selected' pattern {enabled_status}.[/bold cyan]")
    return config['filter_selected_pattern']

def _select_region(message: str) -> Optional[Dict[str, int]]:
    """Helper to let the user draw a box and return the region as a dict."""
    screen_img = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(screen_img), cv2.COLOR_RGB2BGR)
    cv2.namedWindow("Region Selector", cv2.WINDOW_NORMAL)
    console.print(f"[bold cyan]{message}[/bold cyan] (Press Enter to confirm or Esc to cancel)")
    box = cv2.selectROI("Region Selector", img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Region Selector")
    if any(box):
        return {"x": int(box[0]), "y": int(box[1]), "width": int(box[2]), "height": int(box[3])}
    console.print("[yellow]Selection cancelled.[/yellow]")
    return None


def configure_question_region():
    """Interactively select only the question region."""
    global config
    try:
        region = _select_region("Select QUESTION region")
        if region:
            config["question_region"] = region
            config_manager.data = config
            config_manager.save()
            console.print("[bold green]Question region updated.[/bold green]")
    except Exception as e:
        cv2.destroyAllWindows()
        console.print(f"[bold red]Failed to configure question region: {e}[/bold red]")
        logging.error(f"Failed to configure question region: {e}", exc_info=True)


def configure_answer_region(label: str):
    """Interactively select a single answer region given its label."""
    global config
    try:
        region = _select_region(f"Select answer region {label}")
        if region:
            config["answer_regions"][label] = region
            config_manager.data = config
            config_manager.save()
            console.print(f"[bold green]Answer region {label} updated.[/bold green]")
    except Exception as e:
        cv2.destroyAllWindows()
        console.print(f"[bold red]Failed to configure region {label}: {e}[/bold red]")
        logging.error(f"Failed to configure region {label}: {e}", exc_info=True)


def configure_all_regions():
    """Interactively select all question and answer regions."""
    global config
    try:
        region = _select_region("Select QUESTION region")
        if region:
            config["question_region"] = region
        for label in ["A", "B", "C", "D"]:
            region = _select_region(f"Select answer region {label}")
            if region:
                config["answer_regions"][label] = region
        config_manager.data = config
        config_manager.save()
        console.print("[bold green]Regions updated and saved.[/bold green]")
    except Exception as e:
        cv2.destroyAllWindows()
        console.print(f"[bold red]Failed to configure regions: {e}[/bold red]")
        logging.error(f"Failed to configure regions: {e}", exc_info=True)


# Backwards compatibility helper
def configure_regions():
    """Alias for configure_all_regions (legacy name)."""
    configure_all_regions()


def configure_regions_ui():
    """Launch a simple Tkinter UI to draw boxes for question and answer regions."""
    global config
    try:
        screenshot = pyautogui.screenshot()
        width, height = screenshot.size
        root = tk.Tk()
        root.title("Position Configurator")
        root.attributes("-topmost", True)
        canvas = tk.Canvas(root, width=width, height=height)
        canvas.pack()
        tk_img = ImageTk.PhotoImage(screenshot)
        canvas.create_image(0, 0, anchor="nw", image=tk_img)

        steps = [
            ("question_region", "QUESTION"),
            ("A", "Answer A"),
            ("B", "Answer B"),
            ("C", "Answer C"),
            ("D", "Answer D"),
        ]
        idx = 0
        regions: Dict[str, Dict[str, int]] = {}
        rect = None
        start_x = start_y = 0

        info = tk.Label(root, text=f"Draw box for {steps[idx][1]} (Esc to cancel)")
        info.pack()

        def on_press(event):
            nonlocal rect, start_x, start_y
            start_x, start_y = event.x, event.y
            if rect:
                canvas.delete(rect)
            rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline="red", width=2)

        def on_move(event):
            if rect:
                canvas.coords(rect, start_x, start_y, event.x, event.y)

        def on_release(event):
            nonlocal rect, idx
            if not rect:
                return
            x1, y1 = start_x, start_y
            x2, y2 = event.x, event.y
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            regions[steps[idx][0]] = {"x": x, "y": y, "width": w, "height": h}
            canvas.delete(rect)
            rect = None
            idx += 1
            if idx >= len(steps):
                root.quit()
            else:
                info.config(text=f"Draw box for {steps[idx][1]} (Esc to cancel)")

        def cancel(event=None):
            regions.clear()
            root.quit()

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_move)
        canvas.bind("<ButtonRelease-1>", on_release)
        root.bind("<Escape>", cancel)

        root.mainloop()
        root.destroy()

        if len(regions) == len(steps):
            config["question_region"] = regions["question_region"]
            for label in ["A", "B", "C", "D"]:
                config["answer_regions"][label] = regions[label]
            config_manager.data = config
            config_manager.save()
            console.print("[bold green]Regions updated and saved.[/bold green]")
        else:
            console.print("[yellow]Region configuration cancelled.[/yellow]")
    except Exception as e:
        logging.error(f"Failed to configure regions via GUI: {e}", exc_info=True)
        console.print(f"[bold red]Failed to configure regions: {e}[/bold red]")

def configure_regions():
    """Interactively select question and answer regions."""
    global config
    try:
        # Take a fullscreen screenshot once
        screen_img = pyautogui.screenshot()
        img = cv2.cvtColor(np.array(screen_img), cv2.COLOR_RGB2BGR)

        cv2.namedWindow("Region Selector", cv2.WINDOW_NORMAL)

        console.print("[bold cyan]Select QUESTION region and press Enter[/bold cyan]")
        q = cv2.selectROI("Region Selector", img, showCrosshair=True, fromCenter=False)
        if any(q):
            config['question_region'] = {'x': int(q[0]), 'y': int(q[1]), 'width': int(q[2]), 'height': int(q[3])}

        for label in ['A', 'B', 'C', 'D']:
            console.print(f"[bold cyan]Select answer region {label} and press Enter[/bold cyan]")
            r = cv2.selectROI("Region Selector", img, showCrosshair=True, fromCenter=False)
            if any(r):
                config['answer_regions'][label] = {'x': int(r[0]), 'y': int(r[1]), 'width': int(r[2]), 'height': int(r[3])}

        cv2.destroyWindow("Region Selector")


        config_manager.data = config
        config_manager.save()
        console.print("[bold green]Regions updated and saved.[/bold green]")
    except Exception as e:
        cv2.destroyAllWindows()
        console.print(f"[bold red]Failed to configure regions: {e}[/bold red]")
        logging.error(f"Failed to configure regions: {e}", exc_info=True)


def run_accuracy_evaluator_script():
    """Executes the accuracy_evaluator.py script and prints its output."""
    script_path = os.path.join(os.path.dirname(__file__), 'accuracy_evaluator.py')
    if not os.path.exists(script_path):
        print(f"Error: accuracy_evaluator.py not found in the current directory.")
        logging.error("accuracy_evaluator.py not found.")
        return

    print(f"Running {script_path}...")
    try:
        # Use subprocess.run for better control and output capture
        result = subprocess.run(
            [sys.executable, script_path], # Use the same python interpreter
            capture_output=True, 
            text=True, 
            check=True, # Raise exception on non-zero exit code
            encoding='utf-8' # Explicitly set encoding
        )
        print("--- Accuracy Evaluator Output START ---")
        print(result.stdout)
        print("--- Accuracy Evaluator Output END ---")
        if result.stderr:
             print("--- Accuracy Evaluator Errors (stderr) ---")
             print(result.stderr)
             print("-----------------------------------------")
        logging.info(f"Successfully ran accuracy_evaluator.py. Exit code: {result.returncode}")

    except subprocess.CalledProcessError as e:
        print(f"Error running accuracy_evaluator.py: Exit code {e.returncode}")
        print("--- Error Output START ---")
        print(e.stderr or "No standard error output.")
        print(e.stdout or "No standard output.")
        print("--- Error Output END ---")
        logging.error(f"Failed to run accuracy_evaluator.py. Exit code: {e.returncode}", exc_info=False)
        logging.error(f"Stderr:\n{e.stderr}")
        logging.error(f"Stdout:\n{e.stdout}")
        
    except FileNotFoundError:
        print(f"Error: Python interpreter '{sys.executable}' not found or script path incorrect.")
        logging.error(f"Python interpreter '{sys.executable}' not found for running accuracy_evaluator.py")
        
    except Exception as e:
        print(f"An unexpected error occurred while running accuracy_evaluator.py: {e}")
        logging.error(f"Failed to run accuracy_evaluator.py: {e}", exc_info=True)
def show_help():
    """Display help message"""
    print("\n--- Available Commands ---")
    print(" capture / F2   : Capture screen regions, OCR, and find match.")
    print(" autoclick      : Toggle auto-clicking the matched answer region.")
    print(" filterselected : Toggle filtering '[number] selected' pattern from answers.")
    print(" pos            : Configure question and answer regions via GUI.")
    print(" test           : Run the accuracy_evaluator.py script for batch testing.")
    print(" config         : Show current configuration.")
    print(" data <name>: Switch database. Options: default, magic, muggle, all")
    print("                  Example: database magic")
    print(" set <key> <val>: Set a configuration value (e.g., set auto_click True). JSON values need quotes.")
    print("                  Example: set save_all_captured_images True # Save fullscreen image for all captures")
    print(" reload / F3    : Reload configuration and questions data.")
    print(" help           : Show this help message.")
    print(" exit           : Exit the application.")
    print("------------------------")

# Initialize at module level
if __name__ == "__main__":
    # Banner
    console.print("[bold cyan]====================================[/bold cyan]")
    console.print("[bold cyan]     HPMA Quiz Assistant v1.4.2    [/bold cyan]")
    console.print("[bold cyan]====================================[/bold cyan]")
    
    # Initialize the application
    initialize()
    
    # Set up keyboard listener
    print("\nStarting keyboard listener...")
    print("--- Keyboard Shortcuts ---")
    print("F2  - Capture and process quiz")
    print("F3  - Reload configuration")
    print("F10 - Toggle spam capture mode")
    
    print("\n--- Available Commands ---")
    print("test           - Run the accuracy_evaluator.py script for batch testing")
    print("config         - Show current configuration")
    print("pos            - Configure question and answer regions via GUI")
    print("data <name>    - Switch database options (default, magic, muggle, all)")
    print("set <key> <val> - Set configuration values")
    print("help           - Show complete help message")
    print("exit           - Exit the application")
    
    # Start listening for keystrokes
    with keyboard.Listener(on_press=on_press) as listener:
        try:
            # Startup message already shown, full help available with 'help' command
            # show_help()
            
            # Start a complete interactive command loop
            while True:
                try:
                    user_input = input("> ").strip().lower()
                    if not user_input:
                        continue
                        
                    # Split the command and arguments
                    parts = user_input.split()
                    command = parts[0] if parts else ""
                    args = parts[1:] if len(parts) > 1 else []
                    
                    # Process commands
                    if command == "exit":
                        print("Exiting application...")
                        # Stop any active spam capture process
                        if spam_capture_mode:
                            spam_capture_mode = False
                            console.print("[yellow]Stopping spam capture mode...[/yellow]")
                            time.sleep(0.5)  # Give it a moment to clean up
                        
                        # Stop the keyboard listener before exiting
                        listener.stop()
                        break
                    elif command == "help":
                        show_help()
                    elif command == "config":
                        show_config()
                    elif command == "data" and args:
                        switch_database(args[0])
                    elif command == "database" and args:
                        switch_database(args[0])
                    elif command == "autoclick":
                        toggle_auto_click()
                    elif command == "filterselected":
                        toggle_filter_selected()
                    elif command == "pos":
                        configure_regions_ui()
                    elif command == "test":
                        run_accuracy_evaluator_script()
                    elif command == "capture" or command == "f2":
                        capture_and_process()
                    elif command == "reload" or command == "f3":
                        initialize()
                    elif command == "set" and len(args) >= 2:
                        set_config(args)
                    else:
                        show_help()
                except Exception as e:
                    logging.error(f"Error processing command: {e}", exc_info=True)
                    print(f"Error processing command: {e}")
            
            # No need to join here as we've already stopped the listener when exiting
        except Exception as e:
            logging.error(f"Error in keyboard listener: {e}", exc_info=True)

