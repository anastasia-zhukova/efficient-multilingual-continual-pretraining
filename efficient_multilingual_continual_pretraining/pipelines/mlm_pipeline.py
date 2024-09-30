import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertForMaskedLM

from sklearn.model_selection import train_test_split

from efficient_multilingual_continual_pretraining.data import MLMDataset, ChileanMLMDataset
from efficient_multilingual_continual_pretraining.constants import PROJECT_ROOT
from efficient_multilingual_continual_pretraining.models import BaseTrainer
from efficient_multilingual_continual_pretraining import logger


class MLMPipeline:
    @staticmethod
    def run(
        config: dict,
        device: torch.device,
    ):
        task_config = config["task"]
        text_column_name = task_config["text_column_name"]

        if task_config["task_name"] in ["pretrain_german", "pretrain_german_dsir_100k", "atapt_german"]:
            Dataset = MLMDataset
            if task_config["task_name"] == "pretrain_german":
                pd_dataset_dissers = Dataset._load_data(
                    PROJECT_ROOT / "data/mlm/german_dapt/DNB_dissentations",
                    delimiter="\t",
                    text_column_name=text_column_name,
                )
                pd_dataset_manuals = Dataset._load_data(
                    PROJECT_ROOT / "data/mlm/german_dapt/manuals",
                    delimiter="\t",
                    text_column_name=text_column_name,
                )
                pd_dataset_total = pd.concat([pd_dataset_dissers, pd_dataset_manuals], ignore_index=True)[
                    [text_column_name]
                ].head(1_200_000)
            elif task_config["task_name"] == "atapt_german":
                pd_dataset_total = Dataset._load_data(
                    PROJECT_ROOT / "data/mlm/atapt_german",
                    delimiter=",",  # TODO: fix,
                    text_column_name=text_column_name,
                )
            else:
                pd_dataset_total = Dataset._load_data(
                    PROJECT_ROOT / "data/mlm/100k",
                    delimiter="",  # TODO: fix,
                    text_column_name=text_column_name,
                )
        else:
            Dataset = ChileanMLMDataset
            pd_dataset_total = Dataset._load_data(
                PROJECT_ROOT / "data/mlm/chilean_dapt",
                delimiter="",  # TODO: fix
                text_column_name=text_column_name,
            )

        logger.info(f"Total dataset size: {len(pd_dataset_total)}")

        df, dataset_val = train_test_split(
            pd_dataset_total,
            test_size=0.05,
            random_state=config["random_seed"],
        )
        dataset_train, dataset_test = train_test_split(
            pd_dataset_total,
            test_size=0.05,
            random_state=config["random_seed"],
        )
        del df

        dataset_train = Dataset(
            dataset=dataset_train,
            bert_model_name=task_config["model"]["bert_model_name"],
            mlm_probability=task_config["mlm_probability"],
            content_column_name=text_column_name,
        )
        dataset_val = Dataset(
            dataset=dataset_val,
            bert_model_name=task_config["model"]["bert_model_name"],
            mlm_probability=task_config["mlm_probability"],
            content_column_name=text_column_name,
        )
        dataset_test = Dataset(
            dataset=dataset_test,
            bert_model_name=task_config["model"]["bert_model_name"],
            mlm_probability=task_config["mlm_probability"],
            content_column_name=text_column_name,
        )

        dataloader_train = DataLoader(
            dataset_train,
            collate_fn=dataset_train.collate_fn,
            shuffle=True,
            **task_config["dataloader"],
        )
        dataloader_val = DataLoader(
            dataset_val,
            collate_fn=dataset_val.collate_fn,
            shuffle=False,
            **task_config["dataloader"],
        )
        dataloader_test = DataLoader(
            dataset_test,
            collate_fn=dataset_test.collate_fn,
            shuffle=False,
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
            train_dataloader=dataloader_train,
            val_dataloader=dataloader_val,
            test_dataloader=dataloader_test,
            n_steps=task_config["n_steps"],
            steps_to_log=task_config["steps_to_log"],
        )

        torch.save(model.cpu().state_dict(), PROJECT_ROOT / task_config["save_path"])
