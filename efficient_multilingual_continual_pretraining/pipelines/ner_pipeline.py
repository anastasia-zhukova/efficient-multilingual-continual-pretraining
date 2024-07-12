import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from efficient_multilingual_continual_pretraining.constants import PROJECT_ROOT
from efficient_multilingual_continual_pretraining.data import NERDataset
from efficient_multilingual_continual_pretraining.models import BaseTrainer, NERModel


class NERPipeline:
    @staticmethod
    def run(
        config: dict,
        device: torch.device,
    ):
        task_config = config["task"]

        train_dataset = NERDataset(
            data_folder_path=PROJECT_ROOT / f"data/{task_config['task_name']}/train",
            bert_model_name=task_config["model"]["bert_model_name"],
        )
        val_dataset = NERDataset(
            data_folder_path=PROJECT_ROOT / f"data/{task_config['task_name']}/val",
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

        model = NERModel(n_classes=len(train_dataset.entity_mapping), **task_config["model"])
        model = model.to(device)
        optimizer = AdamW(model.parameters(), **task_config["optimizer"])

        trainer = BaseTrainer(
            config["use_watcher"],
            device,
            mode="NER",
            id_to_token_mapping=train_dataset.reverse_entity_mapping,
            criterion=torch.nn.CrossEntropyLoss(),
        )
        trainer.train(
            model,
            optimizer,
            train_dataloader,
            val_dataloader=val_dataloader,
            n_epochs=task_config["n_epochs"],
        )

        torch.save(model.cpu().state_dict(), PROJECT_ROOT / task_config["save_path"])
