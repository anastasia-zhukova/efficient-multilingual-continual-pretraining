from typing import Literal

import torch

from efficient_multilingual_continual_pretraining.models import BaseModel


class RCTModel(BaseModel):
    def __init__(
        self,
        head: torch.nn.Module,
        n_classes: int,
        mode: Literal["finetune", "pretrain"] = "finetune",
        bert_model_name: str = "bert-base-uncased",
        local_files_only: bool = False,
        use_sigmoid_instead_of_softmax: bool = False,
    ) -> None:
        self.n_classes = n_classes
        super(RCTModel, self).__init__(
            head,
            mode,
            bert_model_name,
            local_files_only,
            use_sigmoid_instead_of_softmax=use_sigmoid_instead_of_softmax,
        )

    def forward(
        self,
        input_text,
        paragraph_tokens: list[list[list[int]]],
        cast_to_probabilities: bool = False,
        **kwargs,
    ):
        embeddings = self.bert(**input_text).last_hidden_state

        total_sentences = sum(len(paragraph) for paragraph in paragraph_tokens)
        predictions = torch.zeros((total_sentences, self.n_classes), device=embeddings.device)

        index = 0
        for i, tokens_per_sentence_in_paragraph in enumerate(paragraph_tokens):
            paragraph_embeddings = embeddings[i]
            for sentence_token_indices in tokens_per_sentence_in_paragraph:
                relevant_embeddings = paragraph_embeddings[sentence_token_indices]
                sentence_prediction = self.head(relevant_embeddings.mean(dim=0))
                predictions[index] = sentence_prediction
                index += 1

        if cast_to_probabilities:
            predictions = self.softmax(predictions)

        return predictions