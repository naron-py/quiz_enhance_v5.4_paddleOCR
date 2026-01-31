# HPMA Quiz Assistant v1.4.2 (PaddleOCR Edition)

A high-performance OCR-powered quiz assistant using PaddleOCR for accurate and fast text recognition.

## Features

- ✨ **PaddleOCR Integration** - Fast and accurate OCR (>95% accuracy, <0.3s per region)
- 🚀 **GPU Acceleration** - Leverages CUDA for optimal performance
- 🎯 **TF-IDF Matching** - Intelligent question matching against database
- ⌨️ **Global Hotkeys** - F2 to capture, F3 to clear overlay
- 🔧 **Configurable** - Easy JSON-based configuration

## Quick Start (First Time Setup)

### Prerequisites

- **Windows 10/11**
- **Python 3.11+** ([Download here](https://www.python.org/downloads/))
- **CUDA 11.8** ([Download here](https://developer.nvidia.com/cuda-11-8-0-download-archive))
  - Required for GPU acceleration with PaddlePaddle 2.6.2
  - Select: Windows > x86_64 > 10/11 > exe (network)
  - **Note**: CUDA 11.8 specifically, not newer versions (12.x won't work)
- **CUDA-capable NVIDIA GPU** (GTX 10-series or newer recommended)
- **Git** (for cloning)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/naron-py/quiz_enhance_v5.3_paddleOCR.git
   cd quiz_enhance_v5.3_paddleOCR
   ```

2. **Run the setup script**
   ```bash
   setup_env.bat
   ```
   
   This will:
   - Create a Python virtual environment
   - Install PaddleOCR 2.8.1 and PaddlePaddle GPU 2.6.2
   - Install Torch CPU (to avoid DLL conflicts)
   - Install all other dependencies

3. **Launch the application**
   ```bash
   run_quiz_assistant.bat
   ```

That's it! The app will start with administrator privileges.

## Configuration

Edit `config.json` to customize:

- **Screen regions** - Define question and answer capture areas
- **Hotkeys** - Customize keyboard shortcuts
- **OCR settings** - Toggle debug logs with `show_paddleocr_debug_logs`
- **Database** - Switch between "magic", "muggle", or "all"

### Example Configuration

```json
{
  "question_region": {
    "x": 135,
    "y": 669,
    "width": 678,
    "height": 197
  },
  "show_paddleocr_debug_logs": false,
  "active_database": "magic",
  "require_cuda": true
}
```

## Usage

1. **Start the app** - Run `run_quiz_assistant.bat`
2. **Press F2** - Capture and process quiz question
3. **Press F3** - Clear overlay
4. **Type 'exit'** - Close the application

## Troubleshooting

### "WinError 127" or Import Errors

If you see DLL errors or import issues:

1. **Uninstall conflicting packages**
   ```bash
   venv\Scripts\pip uninstall -y opencv-contrib-python torch paddleocr
   ```

2. **Reinstall in correct order**
   ```bash
   venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
   venv\Scripts\pip install paddlepaddle-gpu==2.6.2
   venv\Scripts\pip install paddleocr==2.8.1
   venv\Scripts\pip install opencv-python
   ```

### No CUDA/GPU

If you don't have a CUDA GPU:
1. Open `config.json`
2. Set `"require_cuda": false`

**Important**: If you have an NVIDIA GPU but see "CUDA not available":
- Make sure you installed **CUDA 11.8** (not 12.x or other versions)
- Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive
- Restart your PC after installation
- Verify installation: Open CMD and run `nvidia-smi`

### Debug Logs

To enable PaddleOCR debug logs:
1. Open `config.json`
2. Set `"show_paddleocr_debug_logs": true`
3. Restart the app

## Technical Details

### Dependencies (Key Versions)

- **PaddlePaddle GPU**: 2.6.2
- **PaddleOCR**: 2.8.1
- **Torch**: 2.10.0+cpu (CPU version to avoid DLL conflicts)
- **OpenCV**: opencv-python (no contrib)

### Why These Versions?

- PaddleOCR 3.x introduced `paddlex` dependency which requires `modelscope` and causes torch DLL conflicts
- PaddleOCR 2.8.1 is the latest stable version compatible with PaddlePaddle 2.6.2
- Torch CPU version satisfies dependencies without Windows DLL issues

## Project Structure

```
quiz_enhance_v5.3_paddleOCR/
├── terminal_app.py          # Main application
├── ocr_processor.py          # PaddleOCR wrapper
├── config_manager.py         # Configuration handler
├── accuracy_evaluator.py     # OCR accuracy testing
├── config.json               # User configuration
├── requirements.txt          # Python dependencies
├── run_quiz_assistant.bat    # Application launcher
├── setup_env.bat            # First-run setup script
├── HPMA_data_magic.csv      # Magic questions database
└── HPMA_data_muggle.csv     # Muggle questions database
```

## Contributing

Feel free to submit issues or pull requests!

## License

MIT License

## Acknowledgments

- **PaddleOCR** - For the excellent OCR library
- **PaddlePaddle** - For the deep learning framework
