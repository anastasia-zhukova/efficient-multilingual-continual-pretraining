import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding, BertTokenizer


# TODO: add behavior without targets if required in the future.
class OpenRepairDataset(Dataset):
    def __init__(
        self,
        positive_pairs: pd.DataFrame,
        negative_pairs: pd.DataFrame,
        bert_model_name: str = "bert-base-uncased",
    ):
        super(OpenRepairDataset, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.positive_pairs = positive_pairs
        self.negative_pairs = negative_pairs

    def __len__(self):
        return len(self.positive_pairs) + len(self.negative_pairs)

    def __getitem__(
        self,
        index: int,
    ):
        is_positive = index < len(self.positive_pairs)
        entry = (
            self.positive_pairs.iloc[index]
            if is_positive
            else self.negative_pairs.iloc[index - len(self.positive_pairs)]
        )
        return entry, is_positive

    def collate_fn(
        self,
        batch_data: list,
    ) -> dict[str, BatchEncoding | torch.Tensor]:
        items_to_encode, targets = zip(*batch_data, strict=False)
        targets = torch.LongTensor(targets)

        questions = [x["problem"] for x in items_to_encode]
        answers = [x["solution"] for x in items_to_encode]

        result = {}
        items_to_encode = [[questions], [answers]]
        item_names = ["question_text", "answer_text"]
        for item_to_encode, item_name in zip(items_to_encode, item_names, strict=True):
            item_encoding = self.tokenizer(
                *item_to_encode,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            result[item_name] = item_encoding

        result["targets"] = targets
        return result
