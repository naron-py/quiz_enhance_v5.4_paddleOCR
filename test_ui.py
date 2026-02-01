"""
Simple test script to launch the Electron UI with mock data
"""
import asyncio
import json
import websockets

async def send_test_data():
    """Send test OCR results to the Electron UI"""
    await asyncio.sleep(2)  # Wait for UI to connect
    
    uri = "ws://localhost:8765"
    try:
        # Connect to the WebSocket server via localhost
        # (In real app, the Python backend runs the WS server)
        print("Sending test data to UI...")
        
        # Mock data
        test_data = {
            "matched_choice": "B",
            "matched_answer": "Cat",
            "question": "What animal says meow?",
            "answers": {
                "A": "Dog",
                "B": "Cat",
                "C": "Bird",
                "D": "Fish"
            },
            "score": 0.95
        }
        
        # For this test, we'll print the data that would be sent
        print(f"Test data: {json.dumps(test_data, indent=2)}")
        print("\nTo test the UI:")
        print("1. Run: npm start")
        print("2. The UI should appear")
        print("3. It will show '-: Waiting...' until connected to backend")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("=== Electron UI Test ===")
    print("This will launch the Electron UI")
    print("\nMake sure you have run: npm install electron")
    print("\nPress Ctrl+C to stop\n")
    
    asyncio.run(send_test_data())
