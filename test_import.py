
import sys
print(f"Python Executable: {sys.executable}")
print(f"Python Path: {sys.path}")

try:
    import paddleocr
    print(f"SUCCESS: PaddleOCR imported from {paddleocr.__file__}")
except ImportError as e:
    print(f"ERROR: {e}")
except Exception as e:
    print(f"UNKNOWN ERROR: {e}")
