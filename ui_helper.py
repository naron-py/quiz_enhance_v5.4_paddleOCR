"""
Helper function to send match results to Electron UI via WebSocket
"""
import logging

logger = logging.getLogger(__name__)

def send_to_ui(matched_choice=None, matched_answer=None, question=None, answers=None, score=0.0):
    """Send OCR match result to Electron UI if UI server is running"""
    try:
        # Use shared instance from ui_server module
        import ui_server as server_module
        
        if server_module.current_server:
            print(f"DEBUG: Found global server instance: {server_module.current_server}")
            server_module.current_server.send_match_result(
                matched_choice=matched_choice or '-',
                matched_answer=matched_answer or 'Waiting...',
                question=question or '-',
                answers=answers or {'A': '-', 'B': '-', 'C': '-', 'D': '-'},
                score=score
            )
        else:
            print(f"DEBUG: No global server instance found in {server_module}")
            
    except Exception as e:
        logger.debug(f"UI not available: {e}")
        print(f"DEBUG: Exception in send_to_ui: {e}")
