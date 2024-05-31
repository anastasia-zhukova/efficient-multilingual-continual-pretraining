from typing import Literal

import torch
from torch import nn
from transformers import BertModel


class BaseModel(nn.Module):
    def __init__(
        self,
        head: torch.nn.Module,
        mode: Literal["finetune", "pretrain"] = "finetune",
        bert_model_name: str = "bert-base-uncased",
        local_files_only: bool = False,
    ) -> None:
        super(BaseModel, self).__init__()

        self.bert = BertModel.from_pretrained(
            bert_model_name,
            local_files_only=local_files_only,
            return_dict=True,
        )

        if mode not in ["pretrain", "finetune"]:
            raise ValueError("The mode for the BERT model should be either 'pretrain' or 'finetune'!")
        self.bert.requires_grad_(mode == "finetune")

        self.head = head
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        input_text,
        cast_to_probabilities: bool = False,
    ):
        embeddings = self.get_embeddings(input_text)
        return self.process_from_embeddings(embeddings, cast_to_probabilities)

    def get_embeddings(
        self,
        input_text: dict,
    ):
        return self.bert(**input_text).last_hidden_state[:, 0, :]

    def process_from_embeddings(
        self,
        embeddings: torch.Tensor,
        cast_to_probabilities: bool = False,
    ):
        head_output = self.head(embeddings)
        return self.softmax(head_output) if cast_to_probabilities else head_output
