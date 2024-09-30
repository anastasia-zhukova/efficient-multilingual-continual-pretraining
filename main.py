import uuid
import warnings

import hydra
import transformers
import wandb
from omegaconf import DictConfig, OmegaConf

from efficient_multilingual_continual_pretraining.pipelines import (
    AmazonReviewsPipeline,
    CaresPipeline,
    NERPipeline,
    OpenRepairPipeline,
    NubesPipeline,
    RCTPipeline,
    ChemProtPipeline,
    MLMPipeline,
)
from efficient_multilingual_continual_pretraining.utils import generate_device, seed_everything


transformers.logging.set_verbosity_error()
warnings.simplefilter("ignore", FutureWarning)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    config = OmegaConf.to_container(config, resolve=True)
    seeds = [config["random_seed"]] if isinstance(config["random_seed"], int) else config["random_seed"]
    print(seeds)
    group_id = str(uuid.uuid4())

    for seed in seeds:
        seed_everything(seed)
        device = generate_device(config["use_cuda_for_train"])

        # WandB handling
        if config["use_watcher"]:
            wandb.init(
                project="efficient_multilingual_continual_pretraining",
                config=config,
                group=group_id,
                job_type=config["task"]["task_name"],
            )

        if config["task"]["task_name"] == "amazon_reviews":
            pipeline = AmazonReviewsPipeline()
        elif config["task"]["task_name"] == "cares":
            pipeline = CaresPipeline()
        elif config["task"]["task_name"] == "openrepair":
            pipeline = OpenRepairPipeline(seed=seed)
        elif config["task"]["task_name"] in ["cantemist", "pharmaconer"]:
            pipeline = NERPipeline()
        elif config["task"]["task_name"] == "nubes":
            pipeline = NubesPipeline()
        elif config["task"]["task_name"] == "rct":
            pipeline = RCTPipeline()
        elif config["task"]["task_name"] == "chemprot":
            pipeline = ChemProtPipeline()
        elif config["task"]["task_name"] == "pretrain_german" or "pretrain_spanish_chilean":
            pipeline = MLMPipeline()
        else:
            raise ValueError("Unsupported pipeline!")

        pipeline.run(config, device)

        if config["use_watcher"]:
            wandb.finish()


if __name__ == "__main__":
    main()
