---
name: quiz-assistant-maintainer
description: Maintain and extend the HPMA Quiz Assistant (PaddleOCR) workflow, including screen capture, OCR preprocessing, TF-IDF matching, auto-clicking, and Electron UI/WebSocket coordination. Use when modifying OCR accuracy, capture regions, matching logic, config defaults, or UI sync behavior in this repo.
---

# HPMA Quiz Assistant Maintenance Skill

## Core workflow map
- Start from `terminal_app.py` for the capture → OCR → match → UI pipeline and hotkeys.
- Use `ocr_processor.py` for OCR initialization, preprocessing, and region OCR batching.
- Use `config_manager.py` for default config keys and persistence behavior.
- Use `ui_server.py` and `ui/app.js` for WebSocket message structure and UI rendering.
- Use `README.md` for setup and known environment constraints (Windows + CUDA).

## Maintain capture and OCR
- Update capture geometry in config defaults or `config.json` and ensure `capture_and_process()` uses the updated regions.
- Adjust OCR preprocessing inside `OCRProcessor.preprocess_image()` for scaling or contrast tuning.
- Keep PaddleOCR initialization aligned with GPU/CPU settings (`require_cuda`, `use_gpu`).

## Maintain matching and auto-click
- Adjust TF-IDF thresholds or answer similarity in `capture_and_process()` and related helpers.
- Ensure auto-click logic remains guarded by configuration and last-click cache.

## Maintain UI sync
- Preserve message schema in `UIServer.send_match_result()` when adjusting UI payloads.
- Mirror any schema changes in `ui/app.js` message handling and display logic.

## Common change patterns
- Add a new config flag:
  - Update defaults in `ConfigManager.default_config()`.
  - Read and apply the flag where behavior is implemented.
- Tune OCR accuracy:
  - Modify preprocessing or image scale factor in `ocr_processor.py`.
- Update UI:
  - Adjust WebSocket payloads in `ui_server.py`.
  - Reflect changes in `ui/app.js`.

## Guardrails
- Avoid breaking hotkey behavior unless also updating the config defaults.
- Keep CUDA guidance consistent with README constraints if changing dependencies.
