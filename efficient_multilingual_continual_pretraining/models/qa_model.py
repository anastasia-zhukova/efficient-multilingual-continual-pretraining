from typing import Literal

import torch

from efficient_multilingual_continual_pretraining.models import BaseModel


class QAModel(BaseModel):
    def __init__(
        self,
        head: torch.nn.Module,
        mode: Literal["finetune", "pretrain"] = "finetune",
        bert_model_name: str = "bert-base-uncased",
        local_files_only: bool = False,
    ) -> None:
        super(QAModel, self).__init__(head, mode, bert_model_name, local_files_only)

    def forward(
        self,
        question_text,
        answer_text,
        cast_to_probabilities: bool = False,
        **kwargs,
    ):
        input_text_embeddings = self.get_embeddings(question_text)
        answer_text_embeddings = self.get_embeddings(answer_text)
        embeddings = torch.cat(
            (input_text_embeddings, answer_text_embeddings),
            dim=1,
        )
        return self.process_from_embeddings(embeddings, cast_to_probabilities)
