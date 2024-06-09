from typing import Literal

import torch

from efficient_multilingual_continual_pretraining.models import BaseModel


class ClassificationModel(BaseModel):
    def __init__(
        self,
        head: torch.nn.Module,
        mode: Literal["finetune", "pretrain"] = "finetune",
        bert_model_name: str = "bert-base-uncased",
        local_files_only: bool = False,
    ) -> None:
        super(ClassificationModel, self).__init__(head, mode, bert_model_name, local_files_only)

    def forward(
        self,
        input_text,
        cast_to_probabilities: bool = False,
        **kwargs,
    ):
        embeddings = self.get_embeddings(input_text)
        return self.process_from_embeddings(embeddings, cast_to_probabilities)
