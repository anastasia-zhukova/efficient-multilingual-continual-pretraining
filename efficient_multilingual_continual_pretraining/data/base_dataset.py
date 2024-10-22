import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding, BertTokenizer


class BaseDataset(Dataset):
    def __init__(
        self,
        object_to_return: pd.DataFrame,
        bert_model_name: str = "bert-base-uncased",
        targets: pd.DataFrame | pd.Series | None = None,
    ):
        super(BaseDataset, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.object_to_return = object_to_return
        self.targets = targets

    def __len__(self):
        return len(self.object_to_return)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[dict, int] | dict:
        if self.targets is not None:
            return self.object_to_return.iloc[index], self.targets.iloc[index]
        return self.object_to_return.iloc[index]

    def collate_fn(
        self,
        batch_data: list,
    ) -> dict[str, BatchEncoding | torch.Tensor]:
        if self.targets is not None:
            items_to_encode, targets = zip(*batch_data, strict=False)
            targets = torch.LongTensor(targets)
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

        result = {"input_text": items_encodings}
        if targets is not None:
            result["targets"] = targets

        return result
