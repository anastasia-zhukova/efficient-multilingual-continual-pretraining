from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from efficient_multilingual_continual_pretraining.constants import PROJECT_ROOT


class MLMDataset(Dataset):
    def __init__(
        self,
        data_folder_path: Path,
        mlm_probability: float = 0.15,
        bert_model_name: str = "bert-base-uncased",
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
        )

        self.data = self._load_data(data_folder_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self,
        index: int,
    ) -> str:
        return self.data.iloc[index]["contents"]

    def collate_fn(
        self,
        batch_data: list[str],
    ) -> dict:
        tokenized_inputs = self.tokenizer(
            batch_data,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        collated = self.data_collator(
            [
                {
                    "input_ids": tokenized_inputs["input_ids"][i],
                    "attention_mask": tokenized_inputs["attention_mask"][i],
                }
                for i in range(len(batch_data))
            ],
        )
        return {
            "input_ids": collated["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": collated["labels"],  # MLM labels
        }

    @staticmethod
    def _load_data(
        data_folder_path: Path,
    ) -> pd.DataFrame:
        dfs = [pd.read_json(file, lines=True) for file in data_folder_path.iterdir()]
        return pd.concat(dfs, ignore_index=True).drop("id", axis=1)


if __name__ == "__main__":
    dataset = MLMDataset(PROJECT_ROOT / "data/mlm")
    print(dataset[0])
    print(dataset.collate_fn([dataset[0], dataset[1]]))
