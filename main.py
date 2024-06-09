import warnings

import hydra
import transformers
from omegaconf import DictConfig, OmegaConf

import wandb
from efficient_multilingual_continual_pretraining.pipelines import (
    AmazonReviewsPipeline,
    CaresPipeline,
    OpenRepairPipeline,
)
from efficient_multilingual_continual_pretraining.utils import generate_device, seed_everything


transformers.logging.set_verbosity_error()
warnings.simplefilter("ignore", FutureWarning)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    config = OmegaConf.to_container(config, resolve=True)
    seed_everything(config["random_seed"])
    device = generate_device(config["use_cuda_for_train"])

    # WandB handling
    if config["use_watcher"]:
        wandb.init(
            project="efficient_multilingual_continual_pretraining",
            config=config,
        )

    if config["task"]["task_name"] == "amazon_reviews":
        pipeline = AmazonReviewsPipeline()
    elif config["task"]["task_name"] == "cares":
        pipeline = CaresPipeline()
    elif config["task"]["task_name"] == "openrepair":
        pipeline = OpenRepairPipeline(seed=config["random_seed"])
    else:
        raise ValueError("Unsupported model type: should be either `bert` or `catboost`!")

    pipeline.run(config, device)

    if config["use_watcher"]:
        wandb.finish()


if __name__ == "__main__":
    main()
