from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BatchEncoding

from efficient_multilingual_continual_pretraining import logger
from efficient_multilingual_continual_pretraining.constants import PROJECT_ROOT


@dataclass
class SingleObject:
    object_name: str
    start: int
    end: int

@dataclass
class RelationshipObject:
    entity_name: str
    participants: list[SingleObject]


class ChemProtDataset(Dataset):
    def __init__(
        self,
        file_path: Path,
        entity_mapping: dict | None = None,
        bert_model_name: str = "bert-base-uncased",
    ):
        super().__init__()

        self.texts, self.relations, self.entity_mapping, self.counts = self._load_data(file_path, entity_mapping)
        self.reverse_entity_mapping = {value: key for key, value in self.entity_mapping.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    def __len__(self):
        return len(self.texts)

    def __getitem__(
        self,
        index: int,
    ) -> (str, list[RelationshipObject]):
        text_id = list(self.texts.keys())[index]
        return self.texts[text_id], self.relations[text_id]

    def collate_fn(
        self,
        batch_data: list[tuple[str, list[RelationshipObject]]],
    ) -> dict[str, BatchEncoding | torch.Tensor]:
        texts, relations = zip(*batch_data, strict=True)

        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        )

        batch_labels = []
        batch_tokens = []
        for i, entry_relations in enumerate(relations):
            entry_labels = []
            entry_tokens = []
            for relation_number, single_relation in enumerate(entry_relations):
                allow_both_append = True
                both_object_tokens = []

                for single_object in single_relation.participants:
                    object_tokens = []
                    debug = []
                    allow_single_append = False
                    for token_index, token in enumerate(tokens["offset_mapping"][i]):
                        if token[0] == 0 and token[1] == 0:
                            continue
                        if token[0] > single_object.end:
                            continue
                        elif token[1] < single_object.start:
                            continue

                        object_tokens.append(token_index)
                        debug.append([token])
                        if token[1] >= single_object.end:
                            allow_single_append = True

                    allow_both_append &= allow_single_append

                    if len(object_tokens) == 0:
                        logger.debug(
                            f"Got zero tokens for relation {relation_number} in paragraph {texts[i]},"
                            f" skipping it as not-covered (most likely truncated due to max length)",
                        )
                        continue

                    if not allow_single_append:
                        logger.debug(
                            f"Append for relation {relation_number} in paragraph {texts[i]} is not allowed"
                            f" as tokens do not cover it fully (most likely truncated due to max length).",
                        )
                        continue

                    both_object_tokens.append(object_tokens)

                if not allow_both_append:
                    logger.debug(
                        f"Append for relation {relation_number} in paragraph {texts[i]} is not allowed"
                        f" as tokens do not cover it fully (most likely truncated due to max length).",
                    )
                    continue

                entry_labels.append(self.entity_mapping[single_relation.entity_name])
                entry_tokens.append(both_object_tokens)

            batch_labels += entry_labels
            batch_tokens.append(entry_tokens)

        labels = torch.LongTensor(batch_labels)

        del tokens["offset_mapping"]
        result = {
            "input_text": tokens,
            "targets": labels,
            "paragraph_tokens": batch_tokens
        }

        return result

    @staticmethod
    def _load_data(
        folder_path: Path,
        mapping: dict | None = None,
    ):
        if mapping is None:
            mapping = {}

        counts = {}

        # Entities handling
        entities = {}
        with (folder_path / "entities.tsv").open(encoding="utf-8") as file:
            for i, line in enumerate(file):
                parts = line.strip().split("\t")
                if len(parts) != 6:
                    logger.error(f"Failed processing line {i}!")
                    raise ValueError(
                        f"Cannot process line {line}: expected 6 parts after a split on tabs, found {len(parts)}!",
                    )
                source_index, entity_number, entity_name, entity_start, entity_end, entity_appeared = parts
                object_instance = SingleObject(entity_name, int(entity_start), int(entity_end))
                if source_index not in entities:
                    entities[source_index] = {entity_number: object_instance}
                else:
                    entities[source_index][entity_number] = object_instance

        # Relations handling
        relations = {}
        with (folder_path / "gold_standard.tsv").open(encoding="utf-8") as file:
            for i, line in enumerate(file):
                parts = line.strip().split("\t")
                if len(parts) != 4:
                    logger.error(f"Failed processing line {i}!")
                    raise ValueError(
                        f"Cannot process line {line}: expected 4 parts after a split on tabs, found {len(parts)}!",
                    )
                source_index, relationship_class, first_object, second_object = parts
                first_object_name = first_object.split(":")[1]
                second_object_name = second_object.split(":")[1]
                relationship_instance = RelationshipObject(
                    relationship_class,
                    [
                        entities[source_index][first_object_name],
                        entities[source_index][second_object_name],
                    ]
                )
                if source_index not in relations:
                    relations[source_index] = [relationship_instance]
                else:
                    relations[source_index].append(relationship_instance)

                if relationship_class not in mapping:
                    mapping[relationship_class] = len(mapping)

                if mapping[relationship_class] not in counts:
                    counts[mapping[relationship_class]] = 1
                else:
                    counts[mapping[relationship_class]] += 1

        # Abstracts handling
        abstracts = {}
        with (folder_path / "abstracts.tsv").open(encoding="utf-8") as file:
            for i, line in enumerate(file):
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    logger.error(f"Failed processing line {i}!")
                    raise ValueError(
                        f"Cannot process line {line}: expected 3 parts after a split on tabs, found {len(parts)}!",
                    )
                if parts[0] not in relations:
                    continue
                abstracts[parts[0]] = parts[1] + " " + parts[2]

        return abstracts, relations, mapping, counts


if __name__ == "__main__":
    # abstracts, relations, mapping = ChemProtDataset._load_data(PROJECT_ROOT / "data/chemprot/train")
    # print(relations["10207608"])
    dataset = ChemProtDataset(PROJECT_ROOT / "data/chemprot/train")
    # print(dataset.entity_mapping)
    # res = dataset.collate_fn([dataset[227]])
    # print(res['targets'])
    # print(dataset[1][1])
    # print(res['input_text'])
    # print(dataset[1])
    # print(res['targets'])
    # print(dataset.entity_mapping)
    # print(res['paragraph_tokens'])
    # print(len(dataset[1][1]))
    # print(res['paragraph_tokens'])
    # print(res['input_text']['offset_mapping'])
    # paragraphs, sentence_types, mapping = RCTDataset._load_data()
    # print(len(paragraphs), len(mapping))
    # print(paragraphs[0][sentence_types[0][3].start:sentence_types[0][3].end])
    print(dataset.counts)
