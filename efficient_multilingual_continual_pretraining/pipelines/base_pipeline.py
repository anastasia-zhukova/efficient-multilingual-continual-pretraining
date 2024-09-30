import torch
from torch import nn

from efficient_multilingual_continual_pretraining.constants import PROJECT_ROOT
from efficient_multilingual_continual_pretraining import logger


class BasePipeline:
    @staticmethod
    def _load_weights(task_config: dict, model: nn.Module) -> None:
        if task_config["bert_weights_path"] is not None:
            logger.info(f"Loading weights from {task_config['bert_weights_path']}")
            weights_path = PROJECT_ROOT / task_config["bert_weights_path"]
            state_dict = torch.load(weights_path)

            filtered_state_dict = {k: v for k, v in state_dict.items() if k.startswith("bert.")}
            filtered_state_dict = {k.replace("bert.", ""): v for k, v in filtered_state_dict.items()}

            model.bert.load_state_dict(filtered_state_dict, strict=False)

            nn.init.xavier_uniform_(model.bert.pooler.dense.weight)
            model.bert.pooler.dense.bias.data.zero_()
