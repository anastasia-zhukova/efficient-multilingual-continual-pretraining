import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BatchEncoding

from efficient_multilingual_continual_pretraining import logger
from efficient_multilingual_continual_pretraining.constants import PROJECT_ROOT


@dataclass
class RCTEntityObject:
    entity_name: str
    start: int
    end: int

    def __iter__(self):
        yield from [self.entity_name, self.start, self.end]


class RCTDataset(Dataset):
    def __init__(
        self,
        file_path: Path,
        entity_mapping: dict | None = None,
        bert_model_name: str = "bert-base-uncased",
    ):
        super().__init__()

        self.texts, self.annotations, self.entity_mapping = self._load_data(file_path, entity_mapping)
        self.reverse_entity_mapping = {value: key for key, value in self.entity_mapping.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    def __len__(self):
        return len(self.texts)

    def __getitem__(
        self,
        index: int,
    ) -> (str, list[RCTEntityObject]):
        return self.texts[index], self.annotations[index]

    def collate_fn(
        self,
        batch_data: list[tuple[str, list[RCTEntityObject]]],
    ) -> dict[str, BatchEncoding | torch.Tensor]:
        texts, entities = zip(*batch_data, strict=True)

        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        )

        total_labels = []
        paragraph_tokens = []
        for i, paragraph_entities in enumerate(entities):
            paragraph_labels = []
            sentences_tokens = []
            for sentence_number, sentence_entity in enumerate(paragraph_entities):
                sentence_tokens = []

                allow_append = False
                for token_index, token in enumerate(tokens["offset_mapping"][i]):
                    if token[0] == 0 and token[1] == 0:
                        continue
                    if token[0] > sentence_entity.end:
                        continue
                    elif token[1] < sentence_entity.start:
                        continue
                    sentence_tokens.append(token_index)
                    if token[1] >= sentence_entity.end:
                        allow_append = True

                if len(sentence_tokens) == 0:
                    logger.debug(
                        f"Got zero tokens for sentence {sentence_number} in paragraph {texts[i]},"
                        f" skipping it as not-covered (most likely truncated due to max length)",
                    )
                    continue

                if not allow_append:
                    logger.debug(
                        f"Append for sentence {sentence_number} in paragraph {texts[i]} is not allowed"
                        f" as tokens do not cover it fully (most likely truncated due to max length).",
                    )
                    continue

                paragraph_labels.append(self.entity_mapping[sentence_entity.entity_name])
                sentences_tokens.append(sentence_tokens)

            paragraph_tokens.append(sentences_tokens)

            total_labels += paragraph_labels

        labels = torch.LongTensor(total_labels)

        del tokens["offset_mapping"]
        result = {
            "input_text": tokens,
            "targets": labels,
            "paragraph_tokens": paragraph_tokens
        }

        return result

    @staticmethod
    def _load_data(
        file_path: Path,
        mapping: dict | None = None,
    ):
        if mapping is None:
            mapping = {}

        with file_path.open(encoding="utf-8") as file:
            paragraphs = []
            sentence_types = []

            current_paragraph = []
            current_sentence_types = []
            offset = 0

            for line in file:
                if line == '\n':
                    logger.debug("Skipping line as empty!")
                    continue

                if line.startswith("#"):
                    paragraphs.append(' '.join(current_paragraph))
                    sentence_types.append(current_sentence_types)

                    logger.debug(
                        f"Reached new paragraph start: {line}. Found {len(current_paragraph)} sentences for previous"
                        f"paragraph. Resetting."
                    )

                    current_paragraph = []
                    current_sentence_types = []
                    offset = 0

                    continue

                parts = line.strip().split("\t")
                if len(parts) != 2:
                    logger.error(current_paragraph)
                    raise ValueError(
                        f"Cannot process line {line}: expected 2 parts after a split on tabs, found {len(parts)}!",
                    )


                sentence_type, sentence = parts
                entity_object = RCTEntityObject(sentence_type, offset, offset + len(sentence) + 1)
                offset += (len(sentence) + 1)
                current_paragraph.append(sentence)
                current_sentence_types.append(entity_object)

                if entity_object.entity_name not in mapping:
                    mapping[entity_object.entity_name] = len(mapping)

            paragraphs.append(''.join(current_paragraph))
            sentence_types.append(current_sentence_types)

        logger.info(f"Finished building dataset at {len(paragraphs)} paragraphs.")

        return paragraphs[1:], sentence_types[1:], mapping

if __name__ == "__main__":
    dataset = RCTDataset(PROJECT_ROOT / "data/rct/debug.txt")
    print(dataset[1])
    print(dataset[1][0][1788:1933])
    res = dataset.collate_fn([dataset[1]])
    # print(res['input_text'])
    # print(dataset[1])
    # print(res['targets'])
    # print(dataset.entity_mapping)
    print(res['paragraph_tokens'])
    # print(len(dataset[1][1]))
    # print(len(res['paragraph_tokens'][0]))
    # print(res['input_text']['offset_mapping'])
    # paragraphs, sentence_types, mapping = RCTDataset._load_data()
    # print(len(paragraphs), len(mapping))
    # print(paragraphs[0][sentence_types[0][3].start:sentence_types[0][3].end])