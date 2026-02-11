from pathlib import Path
import json

class TestDatasetLoader:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self._data = None

    def load(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"Test dataset not found: {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

    def get_data(self):
        if self._data is None:
            self.load()
        return self._data