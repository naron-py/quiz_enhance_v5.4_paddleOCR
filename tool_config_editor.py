import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import pyautogui
import config_manager
import logging
import sys
import os

# Setup simple logging
logging.basicConfig(level=logging.INFO)

def configure_regions_ui():
    """Launch a simple Tkinter UI to draw boxes for question and answer regions."""
    try:
        # Load current config to save to
        config = config_manager.load_config()
        
        screenshot = pyautogui.screenshot()
        width, height = screenshot.size
        
        root = tk.Tk()
        root.title("Position Configurator - HPMA Quiz Assistant")
        root.attributes("-topmost", True)
        # root.state('zoomed') # Maximize on Windows
        
        # Create a container
        frame = tk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(frame, width=width, height=height, cursor="cross")
        canvas.pack(fill=tk.BOTH, expand=True)
        
        tk_img = ImageTk.PhotoImage(screenshot)
        canvas.create_image(0, 0, anchor="nw", image=tk_img)

        steps = [
            ("question_region", "QUESTION Region"),
            ("A", "Answer A"),
            ("B", "Answer B"),
            ("C", "Answer C"),
            ("D", "Answer D"),
        ]
        
        current_step_idx = 0
        regions = {}
        rect = None
        start_x = start_y = 0
        
        # Instructions Label
        lbl_instruction = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="yellow", fg="black")
        lbl_instruction.place(x=10, y=10)

        def update_instruction():
            if current_step_idx < len(steps):
                lbl_instruction.config(text=f"STEP {current_step_idx+1}/{len(steps)}: Draw Box for {steps[current_step_idx][1]} (Esc to Cancel)")
            else:
                lbl_instruction.config(text="Saving...")

        update_instruction()

        def on_press(event):
            nonlocal rect, start_x, start_y
            start_x = canvas.canvasx(event.x)
            start_y = canvas.canvasy(event.y)
            if rect:
                canvas.delete(rect)
            rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline="#00ff00", width=2)

        def on_move(event):
            nonlocal rect
            if rect:
                cur_x = canvas.canvasx(event.x)
                cur_y = canvas.canvasy(event.y)
                canvas.coords(rect, start_x, start_y, cur_x, cur_y)

        def on_release(event):
            nonlocal rect, current_step_idx
            if not rect:
                return
                
            cur_x = canvas.canvasx(event.x)
            cur_y = canvas.canvasy(event.y)
            
            x1 = min(start_x, cur_x)
            y1 = min(start_y, cur_y)
            x2 = max(start_x, cur_x)
            y2 = max(start_y, cur_y)
            
            # Save region
            key = steps[current_step_idx][0]
            regions[key] = {
                "x": int(x1), "y": int(y1), 
                "width": int(x2 - x1), "height": int(y2 - y1)
            }
            
            # Draw permanent rectangle for visual feedback
            canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
            canvas.create_text(x1, y1-10, text=steps[current_step_idx][1], fill="red", anchor="sw", font=("Arial", 10, "bold"))
            
            # Next step
            current_step_idx += 1
            rect = None
            
            if current_step_idx >= len(steps):
                save_and_exit()
            else:
                update_instruction()

        def save_and_exit():
            try:
                config["question_region"] = regions["question_region"]
                for label in ["A", "B", "C", "D"]:
                    config["answer_regions"][label] = regions[label]
                
                config_manager.data = config
                config_manager.save()
                messagebox.showinfo("Success", "Regions updated and saved!\nPlease RELOAD config (F3) in the main app.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save config: {e}")
            finally:
                root.quit()

        def cancel(event=None):
            root.quit()

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_move)
        canvas.bind("<ButtonRelease-1>", on_release)
        root.bind("<Escape>", cancel)
        
        # Maximize after creating
        root.state('zoomed')

        root.mainloop()
        root.destroy()

    except Exception as e:
        print(f"Error launching UI: {e}")
        # If Tkinter fails (e.g. no display), just exit
        pass

if __name__ == "__main__":
    configure_regions_ui()
