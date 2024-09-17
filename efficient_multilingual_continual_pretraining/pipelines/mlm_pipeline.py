import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertForMaskedLM

from efficient_multilingual_continual_pretraining.data import MLMDataset
from efficient_multilingual_continual_pretraining.constants import PROJECT_ROOT
from efficient_multilingual_continual_pretraining.models import BaseTrainer


class MLMPipeline:
    @staticmethod
    def run(
        config: dict,
        device: torch.device,
    ):
        task_config = config["task"]
        dataset = MLMDataset(
            data_folder_path=PROJECT_ROOT / "data/mlm/100k",
            bert_model_name=task_config["model"]["bert_model_name"],
            mlm_probability=task_config["mlm_probability"],
        )

        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            shuffle=True,
            **task_config["dataloader"],
        )

        model = BertForMaskedLM.from_pretrained(
            task_config["model"]["bert_model_name"],
            local_files_only=False,
            return_dict=True,
        )
        model = model.to(device)
        optimizer = AdamW(model.parameters(), **task_config["optimizer"])

        trainer = BaseTrainer(config["use_watcher"], device)
        trainer.pretrain(
            model,
            optimizer,
            dataloader,
            n_epochs=task_config["n_epochs"],
        )

        torch.save(model.cpu().state_dict(), PROJECT_ROOT / task_config["save_path"])
