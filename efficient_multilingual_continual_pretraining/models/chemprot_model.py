from typing import Literal

import torch

from efficient_multilingual_continual_pretraining.models import BaseModel


class ChemProtModel(BaseModel):
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
        super(ChemProtModel, self).__init__(
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

        total_relations = sum(len(relations_list) for relations_list in paragraph_tokens)
        predictions = torch.zeros((total_relations, self.n_classes), device=embeddings.device)

        index = 0
        for i, tokens_per_paragraph in enumerate(paragraph_tokens):
            paragraph_embeddings = embeddings[i]
            for pair_tokens in tokens_per_paragraph:
                object_1_embedding = paragraph_embeddings[pair_tokens[0]].mean(dim=0)
                object_2_embedding = paragraph_embeddings[pair_tokens[1]].mean(dim=0)
                total_embedding = torch.cat([object_1_embedding, object_2_embedding], dim=0)
                sentence_prediction = self.head(total_embedding)
                predictions[index] = sentence_prediction
                index += 1

        if cast_to_probabilities:
            predictions = self.softmax(predictions)

        return predictions
