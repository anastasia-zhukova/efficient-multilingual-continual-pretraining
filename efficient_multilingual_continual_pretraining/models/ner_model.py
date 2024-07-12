from typing import Literal

from torch import nn
from transformers import AutoModel


class NERModel(nn.Module):
    def __init__(
        self,
        bert_model_name: str,
        n_classes: int,
        mode: Literal["finetune", "pretrain"] = "finetune",
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)

        if mode not in ["pretrain", "finetune"]:
            raise ValueError("The mode for the BERT model should be either 'pretrain' or 'finetune'!")
        self.bert.requires_grad_(mode == "finetune")

        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size // 2, self.bert.config.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size // 4, n_classes),
        )

    def forward(
        self,
        input_text,
        **kwargs,
    ):
        outputs = self.bert(**input_text).last_hidden_state
        return self.head(outputs)
