import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BatchEncoding

from efficient_multilingual_continual_pretraining import logger
from efficient_multilingual_continual_pretraining.constants import PROJECT_ROOT


@dataclass
class EntityObject:
    entity_name: str
    start: int
    end: int

    @classmethod
    def from_str(cls, source):
        objects = source.split()
        if len(objects) != 3:
            raise ValueError(f"Could not process source line for EntityObject: {source}!")
        return cls(objects[0], int(objects[1]), int(objects[2]))

    def __iter__(self):
        yield from [self.entity_name, self.start, self.end]


class NERDataset(Dataset):
    def __init__(
        self,
        data_folder_path: Path,
        entity_mapping: dict | None = None,
        bert_model_name: str = "bert-base-uncased",
    ):
        super().__init__()

        self.texts, self.annotations, mapping = self._load_data(data_folder_path)
        self.entity_mapping = entity_mapping if entity_mapping is not None else mapping
        self.reverse_entity_mapping = {value: key for key, value in self.entity_mapping.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    def __len__(self):
        return len(self.texts)

    def __getitem__(
        self,
        index: int,
    ) -> (str, list[EntityObject]):
        return self.texts[index], self.annotations[index]

    def collate_fn(
        self,
        batch_data: list[tuple[str, list[EntityObject]]],
    ) -> dict[str, BatchEncoding | torch.Tensor]:
        texts, annotations = zip(*batch_data, strict=True)

        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        )

        # Targets processing
        result_labels = []
        for i in range(len(batch_data)):
            labels_out = [0] * len(tokens["input_ids"][i])
            for token_index, (token_start, token_end) in enumerate(tokens["offset_mapping"][i]):
                if token_start == 0 and token_end == 0:
                    labels_out[token_index] = -100  # Special tokens

                # TODO: might not work correctly, it would be good to ensure it always works.
                for annotation_tag, annotation_start, annotation_end in annotations[i]:
                    if token_end > annotation_end:
                        continue
                    elif annotation_start == token_start:
                        labels_out[token_index] = self.entity_mapping[f"B-{annotation_tag}"]
                        break
                    elif annotation_start < token_start:
                        labels_out[token_index] = self.entity_mapping[f"I-{annotation_tag}"]
                        break

            result_labels.append(labels_out)

        del tokens["offset_mapping"]
        result = {
            "input_text": tokens,
            "targets": torch.LongTensor(result_labels),
        }

        return result

    @staticmethod
    def _load_data(base_path: Path):
        texts_dir = base_path / "texts"
        annotations_dir = base_path / "annotations"

        n_texts_files = len(os.listdir(texts_dir))
        n_annotations_files = len(os.listdir(annotations_dir))
        if n_texts_files != n_annotations_files:
            raise ValueError(f"Inconsistent number of texts and annotations: {n_texts_files} vs {n_annotations_files}!")

        texts = []
        annotations = []
        mapping = {"O": 0}

        for text_file in texts_dir.iterdir():
            annotation_file_name = text_file.with_suffix(".ann").name
            with text_file.open() as file:
                texts.append(file.read())

            annotations_per_line = []
            with (annotations_dir / annotation_file_name).open() as file:
                for line in file:
                    if line.startswith("#"):
                        logger.debug(
                            f"Skipping line {line} for file"
                            f" {annotations_dir / annotation_file_name} as a commented-out one.",
                        )
                        continue

                    parts = line.strip().split("\t")
                    if len(parts) != 3:
                        raise ValueError(
                            f"Cannot process line {line}: expected 3 parts after a split on tabs, found {len(parts)}!",
                        )

                    annotation = EntityObject.from_str(parts[1])
                    if f"B-{annotation.entity_name}" not in mapping:
                        mapping[f"B-{annotation.entity_name}"] = len(mapping)
                        mapping[f"I-{annotation.entity_name}"] = len(mapping)

                    annotations_per_line.append(annotation)

            annotations.append(annotations_per_line)

        return texts, annotations, mapping


if __name__ == "__main__":
    dataset = NERDataset(PROJECT_ROOT / "data/cantemist/test", bert_model_name="IIC/BETO_Galen")
    print((dataset.collate_fn([dataset[0], dataset[1], dataset[3]])["labels"] > 0).sum(dim=1))
    print(dataset[2][1])
    print(len(dataset[2][1]))
    print(dataset[0])
