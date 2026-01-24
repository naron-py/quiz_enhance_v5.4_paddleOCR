import json
import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ConfigManager:
    """Load and save configuration values from a JSON file."""

    path: str = "config.json"
    _data: Dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.load()

    @property
    def data(self) -> Dict:
        """Access the configuration data.

        Using a property ensures attribute access works even if the
        underlying data dictionary hasn't been created yet, preventing
        ``AttributeError`` during import time in consumers.
        """
        return self._data

    @data.setter
    def data(self, value: Dict) -> None:
        """Replace the configuration data.

        Using a setter allows callers to assign a new configuration
        dictionary to the manager.  The provided value completely
        replaces the existing configuration, so callers should ensure
        any required keys are present before saving.
        """
        self._data = value

    def default_config(self) -> Dict:
        return {
            'question_region': {'x': 115, 'y': 680, 'width': 680, 'height': 200},
            'answer_regions': {
                'A': {'x': 390, 'y': 880, 'width': 550, 'height': 90},
                'B': {'x': 1040, 'y': 880, 'width': 550, 'height': 90},
                'C': {'x': 390, 'y': 1000, 'width': 550, 'height': 90},
                'D': {'x': 1040, 'y': 1000, 'width': 550, 'height': 90}
            },
            'match_mode': 'Classic',
            'show_processing_times': True,
            'auto_click': True,
            'show_ocr_answer_choices_terminal': True,
            'capture_fullscreen_on_nomatch': True,
            'filter_answer_choice_tags': False,
            'save_all_captured_images': False,
            'active_database': 'default',
            'filter_selected_pattern': True,
            'log_level': 'INFO',
            'image_scale_factor': 1.0,
            'require_cuda': False,
            'tfidf_cache_enabled': True,
            'tfidf_cache_dir': '.cache',
            'hotkeys': {
                'capture': 'f2',
                'reload': 'f3',
                'autoclick': 'f9',
                'autoscan': 'f10'
            },
            'answer_similarity_close_margin': 0.05,
            'self_test_cases': []
        }

    def load(self) -> Dict:
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                self._data = json.load(f)
        else:
            self._data = self.default_config()
            self.save()

        # Ensure defaults for any missing fields
        defaults = self.default_config()
        for key, value in defaults.items():
            if key not in self._data:
                self._data[key] = value
        return self._data

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self._data, f, indent=4)
