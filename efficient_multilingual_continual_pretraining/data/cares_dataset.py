import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding, BertTokenizer


class CaresDataset(Dataset):
    def __init__(
        self,
        object_to_return: pd.DataFrame,
        total_classes: int | None = None,
        bert_model_name: str = "bert-base-uncased",
        targets: pd.DataFrame | pd.Series | None = None,
    ):
        super(CaresDataset, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.object_to_return = object_to_return
        self.total_classes = total_classes
        self.targets = targets

        if (self.targets is not None) != (self.total_classes is not None):
            raise ValueError("Mismatch: either targets provided and total classes is not, or the other way around!")

    def __getitem__(
        self,
        index: int,
    ) -> tuple[dict, int] | dict:
        if self.targets is not None:
            return self.object_to_return.iloc[index], self.targets.iloc[index]
        return self.object_to_return.iloc[index]

    def __len__(self):
        return len(self.object_to_return)

    def collate_fn(self, batch_data: list) -> BatchEncoding | tuple[BatchEncoding, torch.Tensor]:
        if self.targets is not None:
            items_to_encode, raw_targets = zip(*batch_data, strict=False)
            targets = torch.zeros((len(raw_targets), self.total_classes))
            for i, items in enumerate(raw_targets):
                targets[i][items] = 1
        else:
            items_to_encode = batch_data
            targets = None

        result = []
        for key in items_to_encode[0].keys():
            result.append([x[key] for x in items_to_encode])

        items_encodings = self.tokenizer(
            *result,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if targets is not None:
            return items_encodings, targets
        return items_encodings
