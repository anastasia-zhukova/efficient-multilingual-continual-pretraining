import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from efficient_multilingual_continual_pretraining.constants import PROJECT_ROOT
from efficient_multilingual_continual_pretraining.data import BaseDataset
from efficient_multilingual_continual_pretraining.models import BaseTrainer, ClassificationModel


class AmazonReviewsPipeline:
    @classmethod
    def run(
        cls,
        config: dict,
        device: torch.device,
    ):
        task_config = config["task"]

        train_dataframe = cls._clean_df(pd.read_csv(PROJECT_ROOT / "data/amazon_reviews_multi/train.csv"))
        val_dataframe = cls._clean_df(pd.read_csv(PROJECT_ROOT / "data/amazon_reviews_multi/validation.csv"))

        train_dataset = BaseDataset(
            train_dataframe.drop("stars", axis=1),
            bert_model_name=task_config["model"]["bert_model_name"],
            targets=train_dataframe["stars"],
        )
        val_dataset = BaseDataset(
            val_dataframe.drop("stars", axis=1),
            bert_model_name=task_config["model"]["bert_model_name"],
            targets=val_dataframe["stars"],
        )

        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=train_dataset.collate_fn,
            shuffle=True,
            **task_config["dataloader"],
        )
        val_dataloader = DataLoader(
            val_dataset,
            collate_fn=val_dataset.collate_fn,
            shuffle=False,
            **task_config["dataloader"],
        )

        # TODO: put to config?
        hidden_size = 768
        num_labels = 5
        model_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_labels),
        )

        model = ClassificationModel(model_head, **task_config["model"])

        # Use pretrained weights
        weights_path = PROJECT_ROOT / "model_weights/deepset_gbert_base_pretrained_100.pth"
        state_dict = torch.load(weights_path)
        filtered_state_dict = {k: v for k, v in state_dict.items() if k.startswith("bert.")}
        filtered_state_dict = {k.replace("bert.", ""): v for k, v in filtered_state_dict.items()}
        model.bert.load_state_dict(filtered_state_dict, strict=False)
        nn.init.xavier_uniform_(model.bert.pooler.dense.weight)
        model.bert.pooler.dense.bias.data.zero_()

        model = model.to(device)
        optimizer = AdamW(model.parameters(), **task_config["optimizer"])

        trainer = BaseTrainer(config["use_watcher"], device, mode="multi-class", n_classes=num_labels)
        trainer.train(
            model,
            optimizer,
            train_dataloader,
            val_dataloader=val_dataloader,
            n_epochs=task_config["n_epochs"],
        )

        torch.save(model.cpu().state_dict(), PROJECT_ROOT / task_config["save_path"])

    # TODO: add other categories falling under electronics as well.
    @staticmethod
    def _clean_df(dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe[(dataframe["language"] == "de") & (dataframe["product_category"] == "electronics")]
        dataframe = dataframe.drop(["review_id", "product_id", "reviewer_id", "language", "product_category"], axis=1)
        dataframe["stars"] -= 1
        dataframe = dataframe[["review_body", "review_title", "stars"]]
        return dataframe
