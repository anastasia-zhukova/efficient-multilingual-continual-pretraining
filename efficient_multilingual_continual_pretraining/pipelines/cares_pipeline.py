import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from efficient_multilingual_continual_pretraining.constants import PROJECT_ROOT
from efficient_multilingual_continual_pretraining.data import CaresDataset
from efficient_multilingual_continual_pretraining.models import BaseModel, Trainer


class CaresPipeline:
    def __init__(self):
        self.mapping = {}

    def run(
        self,
        config: dict,
        device: torch.device,
    ):
        task_config = config["task"]

        train_dataframe = self._clean_df(pd.read_parquet(PROJECT_ROOT / "data/cares/train.parquet"))
        val_dataframe = self._clean_df(pd.read_parquet(PROJECT_ROOT / "data/cares/test.parquet"))

        total_classes = len(self.mapping)

        train_dataset = CaresDataset(
            train_dataframe.drop("general", axis=1),
            total_classes=total_classes,
            bert_model_name=task_config["model"]["bert_model_name"],
            targets=train_dataframe["general"],
        )
        val_dataset = CaresDataset(
            val_dataframe.drop("general", axis=1),
            total_classes=total_classes,
            bert_model_name=task_config["model"]["bert_model_name"],
            targets=val_dataframe["general"],
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
        model_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, total_classes),
        )

        model = BaseModel(model_head, **task_config["model"])
        model = model.to(device)
        optimizer = AdamW(model.parameters(), **task_config["optimizer"])

        trainer = Trainer(config["use_watcher"], device, mode="multi-label", criterion=torch.nn.BCELoss())
        trainer.train(
            model,
            optimizer,
            train_dataloader,
            val_dataloader=val_dataloader,
            n_epochs=task_config["n_epochs"],
        )

        torch.save(model.cpu().state_dict(), PROJECT_ROOT / task_config["save_path"])

    def _clean_df(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        dataframe = dataframe[["full_text", "general"]]
        dataframe.loc[:, "general"] = dataframe["general"].apply(self._process_line)
        return dataframe

    def _process_line(self, line_array):
        return [self.mapping.setdefault(category, len(self.mapping)) for category in line_array]
