import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from efficient_multilingual_continual_pretraining.constants import PROJECT_ROOT
from efficient_multilingual_continual_pretraining.data import ChemProtDataset
from efficient_multilingual_continual_pretraining.models import BaseTrainer, ChemProtModel


class ChemProtPipeline:
    @classmethod
    def run(
        cls,
        config: dict,
        device: torch.device,
    ):
        task_config = config["task"]

        train_dataset = ChemProtDataset(
            file_path=PROJECT_ROOT / f"data/{task_config['task_name']}/train",
            bert_model_name=task_config["model"]["bert_model_name"],
        )
        val_dataset = ChemProtDataset(
            file_path=PROJECT_ROOT / f"data/{task_config['task_name']}/val",
            entity_mapping=train_dataset.entity_mapping,
            bert_model_name=task_config["model"]["bert_model_name"],
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
        num_labels = len(train_dataset.entity_mapping)
        model_head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_labels),
        )

        model = ChemProtModel(model_head, num_labels, **task_config["model"])
        model = model.to(device)
        optimizer = AdamW(model.parameters(), **task_config["optimizer"])

        trainer = BaseTrainer(
            config["use_watcher"],
            device,
            mode="multi-class",
            n_classes=num_labels,
            criterion=nn.CrossEntropyLoss(weight=torch.tensor([3, 1, 12, 9, 3], dtype=torch.float32)).to(device)
        )
        trainer.train(
            model,
            optimizer,
            train_dataloader,
            val_dataloader=val_dataloader,
            n_epochs=task_config["n_epochs"],
        )

        torch.save(model.cpu().state_dict(), PROJECT_ROOT / task_config["save_path"])
