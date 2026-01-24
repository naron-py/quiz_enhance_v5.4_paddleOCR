import json
import threading
from pathlib import Path

import terminal_app as core

try:
    import webview
except ImportError:
    webview = None


class QuizAPI:
    def __init__(self):
        self.status = "SYSTEM READY"
        self.subtitle = "v2.4.0 - DATABASE SYNCED"
        self._lock = threading.Lock()

    def _build_match(self, question_text, choices):
        if not question_text or core.questions_df is None or core.tfidf_vectorizer is None:
            return {}

        tfidf_threshold = core.config.get("tfidf_threshold", 0.85)
        matching_entries = core.find_all_matching_questions(
            question_text,
            core.questions_df,
            core.tfidf_vectorizer,
            core.tfidf_matrix,
            threshold=tfidf_threshold,
        )

        top_entry = matching_entries[0]
        answer_similarity_threshold = core.config.get("answer_similarity_threshold", 0.7)
        best = core.select_best_answer_choice(matching_entries, choices, answer_similarity_threshold)

        if best and best.get("similarity", 0.0) >= answer_similarity_threshold:
            return {
                "best_choice": best.get("choice"),
                "match_answer": best.get("entry", {}).get("answer", ""),
                "match_question": best.get("entry", {}).get("question", ""),
                "match_score": f"{best.get('similarity', 0.0):.3f}",
            }

        return {
            "best_choice": "",
            "match_answer": top_entry.get("answer", ""),
            "match_question": top_entry.get("question", ""),
            "match_score": "",
        }

    def get_state(self):
        source = core.last_ocr_snapshot if getattr(core, "last_ocr_snapshot", None) else core.recognized_text
        question_text = source.get("question", "") if source else ""
        choices = {label: source.get(label, "") for label in ["A", "B", "C", "D"]}

        match = self._build_match(question_text, choices)
        timings = core.timings or {}
        timing_text = " | ".join(
            f"{key.upper()}: {value:.3f}s" for key, value in timings.items()
        ) if timings else "No timings recorded."

        state = {
            "ok": True,
            "status": self.status,
            "subtitle": self.subtitle,
            "question_index": core.question_capture_count,
            "question": question_text,
            "choices": choices,
            "timings": timing_text,
            "best_choice": match.get("best_choice"),
            "match_answer": match.get("match_answer"),
            "match_question": match.get("match_question"),
            "match_score": match.get("match_score"),
            "autoclick": core.auto_click,
            "autoscan": core.spam_capture_mode,
        }
        return state

    def run_action(self, action):
        with self._lock:
            try:
                if action == "capture":
                    self.status = "CAPTURING"
                    core.capture_and_process()
                elif action == "reload":
                    self.status = "RELOADING"
                    core.initialize()
                elif action == "autoclick":
                    core.toggle_auto_click()
                    self.status = "AUTOCLICK ON" if core.auto_click else "AUTOCLICK OFF"
                elif action == "autoscan":
                    core.toggle_spam_capture_mode()
                    self.status = "AUTOSCAN ON" if core.spam_capture_mode else "AUTOSCAN OFF"
                elif action == "config":
                    return {"status": "CONFIG", "data": json.dumps(core.config, indent=2)}
                elif action == "pos":
                    core.configure_regions_ui()
                    self.status = "REGIONS UPDATED"
                elif action.startswith("data:"):
                    _, db_name = action.split(":", 1)
                    ok = core.switch_database(db_name.strip())
                    self.status = "DATABASE OK" if ok else "DATABASE FAILED"
                elif action == "data":
                    return {"status": "DATA", "prompt": "Enter database: default, magic, muggle, all"}
                elif action.startswith("set:"):
                    _, key, val = action.split(":", 2)
                    core.set_config([key, val])
                    self.status = f"SET {key}"
                elif action == "set":
                    return {"status": "SET", "prompt": "Enter key and value"}
                elif action == "test":
                    core.run_accuracy_evaluator_script()
                    self.status = "TEST COMPLETE"
                elif action == "selftest":
                    core.run_self_test()
                    self.status = "SELFTEST COMPLETE"
                elif action == "help":
                    return {
                        "status": "HELP",
                        "data": (
                            "capture, reload, autoclick, autoscan, config, pos, data, set, "
                            "test, selftest, exit"
                        ),
                    }
                elif action == "exit":
                    self.status = "EXITING"
                    if webview is not None:
                        webview.destroy_window()
                else:
                    self.status = "UNKNOWN ACTION"
            except Exception as exc:
                self.status = f"ERROR: {exc}"

        return {"status": self.status}


def main():
    if webview is None:
        raise SystemExit("pywebview is not installed. Run: pip install pywebview")

    core.initialize()

    html_path = Path(__file__).parent / "ui_web" / "index.html"
    api = QuizAPI()

    def on_press(key):
        try:
            if core.hotkeys.get("capture", {}).get("key") == key:
                threading.Thread(target=api.run_action, args=("capture",), daemon=True).start()
            elif core.hotkeys.get("reload", {}).get("key") == key:
                threading.Thread(target=api.run_action, args=("reload",), daemon=True).start()
            elif core.hotkeys.get("autoclick", {}).get("key") == key:
                threading.Thread(target=api.run_action, args=("autoclick",), daemon=True).start()
            elif core.hotkeys.get("autoscan", {}).get("key") == key:
                threading.Thread(target=api.run_action, args=("autoscan",), daemon=True).start()
        except Exception:
            pass

    hotkey_listener = core.keyboard.Listener(on_press=on_press)
    hotkey_listener.daemon = True
    hotkey_listener.start()

    window = webview.create_window(
        "HPMA Quiz Assistant UI",
        html_path.as_uri(),
        js_api=api,
        width=1200,
        height=900,
    )
    webview.start()


if __name__ == "__main__":
    main()
