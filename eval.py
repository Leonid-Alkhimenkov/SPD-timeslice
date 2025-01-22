# Evaluation script
# Example of usage:
# CUDA_VISIBLE_DEVICES=0 python eval.py --model_dir "path/to/logs/experiment_logs/TrackEmbedder/version_16" --device cuda

import os
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

from src.logging_utils import setup_logger
from src.evaluation import ModelEvaluator
from torch import nn
from typing import Callable, Optional
from absl import flags
from absl import app
import pytorch_lightning as pl
import logging
import gin

from src.model import TrackEmbedder
from src.training import TripletType, DistanceType
from src.clustering import clustering_algorithm
from src.transformations import ConstraintsNormalizer


FLAGS = flags.FLAGS
flags.DEFINE_string(
    name='model_dir', default=None,
    help='Path to the directory with saved model.'
)
flags.DEFINE_string(
    name='device', default="cpu",
    help='Device on which to run evaluation.'
)
flags.DEFINE_enum(
    name='log', default='INFO',
    enum_values=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'],
    help='Level of logging'
)


LOGGER = logging.getLogger("eval")


@gin.configurable
def experiment(
        model_dir: str,
        model: nn.Module = gin.REQUIRED,
        random_seed: int = gin.REQUIRED,
        test_samples: int = gin.REQUIRED,
        detector_efficiency: float = gin.REQUIRED,
        hits_normalizer: Optional[Callable] = gin.REQUIRED,
        num_workers: int = gin.REQUIRED,
        pin_memory: bool = gin.REQUIRED,
        device: str = "cpu",
        **kwargs # for the rest of unnecessary configs
):
    setup_logger(LOGGER, model_dir, stage="eval")

    LOGGER.info(
        "GOT config: \n======config======\n "
        f"{gin.config_str()} "
        "\n========config======="
    )

    LOGGER.info(f"Setting random seed to {random_seed}")
    pl.seed_everything(random_seed)

    LOGGER.info(f"Start evaluation.")
    evaluator = ModelEvaluator(
        model_dir=model_dir,
        model=model,
        clustering_model=clustering_algorithm(),
        test_samples=test_samples,
        detector_efficiency=detector_efficiency,
        hits_normalizer=hits_normalizer,
        num_workers=num_workers,
        pin_memory=pin_memory,
        device=device
    )

    evaluator.visualize_embeddings()

    result = evaluator.evaluate(save_result=True)
    LOGGER.info(result)
    LOGGER.info("End of evaluation")


def main(argv):
    del argv
    config_path = os.path.join(FLAGS.model_dir, "train_config.cfg")
    gin.parse_config(open(config_path))
    LOGGER.setLevel(FLAGS.log)
    experiment(FLAGS.model_dir, device=FLAGS.device)


if __name__ == "__main__":
    app.run(main)
