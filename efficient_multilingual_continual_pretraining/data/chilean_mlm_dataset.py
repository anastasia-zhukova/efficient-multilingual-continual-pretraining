from pathlib import Path

import pandas as pd

from efficient_multilingual_continual_pretraining.data import MLMDataset


class ChileanMLMDataset(MLMDataset):
    @staticmethod
    def _load_data(
        data_folder_path: Path,
        text_column_name: str,
        delimiter: str,
    ) -> pd.DataFrame:

        texts = []
        for text_file in data_folder_path.iterdir():
            with text_file.open(encoding="utf-8") as file:
                texts.append(file.read())

        return pd.DataFrame({"text": texts})
