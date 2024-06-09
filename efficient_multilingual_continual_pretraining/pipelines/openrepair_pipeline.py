import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from efficient_multilingual_continual_pretraining import logger
from efficient_multilingual_continual_pretraining.constants import PROJECT_ROOT
from efficient_multilingual_continual_pretraining.data import OpenRepairDataset
from efficient_multilingual_continual_pretraining.models import BaseTrainer, QAModel
from efficient_multilingual_continual_pretraining.utils import log_with_message


class OpenRepairPipeline:
    def __init__(
        self,
        seed: int,
    ):
        self.seed = seed
        self.random_generator = np.random.default_rng(seed)

    def run(
        self,
        config: dict,
        device: torch.device,
    ):
        task_config = config["task"]

        positive_pairs = self._clean_df(pd.read_csv(PROJECT_ROOT / "data/OpenRepairData_v0.csv", encoding="iso-8859-1"))
        train_positive_pairs, train_negative_pairs, val_positive_pairs, val_negative_pairs = self._generate_data(
            positive_pairs,
            task_config["data"]["n_negative"],
        )

        train_dataset = OpenRepairDataset(
            train_positive_pairs,
            train_negative_pairs,
            bert_model_name=task_config["model"]["bert_model_name"],
        )
        val_dataset = OpenRepairDataset(
            val_positive_pairs,
            val_negative_pairs,
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
        num_labels = 2
        head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_labels),
        )

        model = QAModel(head, **task_config["model"])
        model = model.to(device)
        optimizer = AdamW(model.parameters(), **task_config["optimizer"])
        criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([5, 1])).to(device)

        trainer = BaseTrainer(config["use_watcher"], device, mode="binary", criterion=criterion)
        trainer.train(
            model,
            optimizer,
            train_dataloader,
            val_dataloader=val_dataloader,
            n_epochs=task_config["n_epochs"],
        )

        torch.save(model.cpu().state_dict(), PROJECT_ROOT / task_config["save_path"])

    @log_with_message("generating data")
    def _generate_data(
        self,
        positive_pairs: pd.DataFrame,
        n_negative: int,
    ):
        sampled_indexes = self._sample_objects(positive_pairs, n_negative, self.random_generator)
        question_repeated = np.repeat(positive_pairs["problem"].values, sampled_indexes.shape[1])
        answers_indices = sampled_indexes.flatten()
        answers_selected = positive_pairs["solution"].values[answers_indices]
        negative_pairs = pd.DataFrame(
            {
                "problem": question_repeated,
                "solution": answers_selected,
            },
        )

        train_positive_pairs, val_positive_pairs = train_test_split(positive_pairs, random_state=self.seed)
        train_negative_pairs, val_negative_pairs = train_test_split(negative_pairs, random_state=self.seed)

        return train_positive_pairs, train_negative_pairs, val_positive_pairs, val_negative_pairs

    @staticmethod
    def _sample_objects(
        positive_pairs: pd.DataFrame,
        n_negative: int,
        random_generator: np.random.Generator,
    ):
        # Assuming that the items in pairs are all unique, i.e. there is one unique problem and one unique solution.
        allowed_matrix = np.ones([len(positive_pairs), len(positive_pairs)]) - np.eye(len(positive_pairs))
        probability_matrix = random_generator.random(allowed_matrix.shape)
        indices = np.argsort(probability_matrix * allowed_matrix, axis=1)[:, -n_negative:]
        return indices

    @staticmethod
    def _clean_df(dataframe: pd.DataFrame) -> pd.DataFrame:
        # TBD: I believe there is no point in using the category since the task is not to maximize the overall
        #   score rather than to maximize the studied method's score over the initial score.
        dataframe = dataframe[["Problem_refined", "Solution_refined"]]
        original_size = len(dataframe)
        dataframe = dataframe.dropna().reset_index(drop=True)
        logger.debug(
            f"Dropped {original_size - len(dataframe)} rows with NaN values, have {len(dataframe)} entries remaining.",
        )
        dataframe.columns = ["problem", "solution"]

        return dataframe
