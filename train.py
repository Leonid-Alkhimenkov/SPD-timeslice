import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import matplotlib
matplotlib.use('Agg')

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import gin
import logging
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm 
from absl import flags
from absl import app
from torch import nn
import torch
from typing import Optional, Callable
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.metrics import SilhouetteScoreMetric
from src.transformations import ConstraintsNormalizer
from src.model import TrackEmbedder
from src.dataset import time_slice_collator, SPDTimesliceTracksDataset, DatasetMode
from src.training import TripletTracksEmbedder, TripletType, DistanceType
from src.logging_utils import setup_logger
from src.clustering import clustering_algorithm
from src.regression import VertexRegressor
from src.data_generation import SPDEventGenerator


FLAGS = flags.FLAGS
flags.DEFINE_string(
    name='config', default=None,
    help='Path to the config file to use.'
)
flags.DEFINE_enum(
    name='log', default='INFO',
    enum_values=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'],
    help='Level of logging'
)

BATCH_SIZE = 1  # one timeslice in a batch

LOGGER = logging.getLogger("train")


@gin.configurable
def experiment(
        model: nn.Module,
        logging_dir: str = "experiment_logs",
        random_seed: int = 42,
        num_epochs: int = 10,
        train_samples: int = 10000,
        test_samples: int = 1000,
        detector_efficiency: float = 1.0,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-2,
        triplet_margin: float = 0.1,
        type_of_triplets: TripletType = TripletType.semihard,
        distance: DistanceType = DistanceType.euclidean_distance,
        hits_normalizer: Optional[Callable] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        regression_samples: int = 100000,
        metrics: list = [SilhouetteScoreMetric],
):
    os.makedirs(logging_dir, exist_ok=True)
    tb_logger = TensorBoardLogger(logging_dir, name=model.__class__.__name__)
    setup_logger(LOGGER, tb_logger.log_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{tb_logger.log_dir}",
        filename=f'{{epoch}}-{{step}}'
    )

    with open(os.path.join(tb_logger.log_dir, "train_config.cfg"), "w") as f:
        f.write(gin.config_str())

    LOGGER.info(f"Log directory {tb_logger.log_dir}")
    LOGGER.info(
        "GOT config: \n======config======\n "
        f"{gin.config_str()} "
        "\n========config======="
    )

    LOGGER.info(f"Setting random seed to {random_seed}")
    pl.seed_everything(random_seed)

    LOGGER.info("Preparing datasets for regression")
    spd_gen = SPDEventGenerator(detector_eff=detector_efficiency)
    np.random.seed(random_seed)

    tracks, vertices = [], []

    for i in tqdm(range(regression_samples), desc="Generating regression data"):
        event = spd_gen.generate_spd_event()
        unique_track_ids = np.unique(event.track_ids)
        
        for track_id in unique_track_ids:
            track_mask = event.track_ids == track_id
            track_hits = event.hits[track_mask]
            # print(f"track_hits.shape - {track_hits.shape}")
            tracks.append(np.array(track_hits))
            vertices.append(event.vertex)

    tracks = np.array(tracks, dtype=object)
    vertices = np.array(vertices, dtype=np.float32)
    X_train, X_val, y_train, y_val = train_test_split(tracks, vertices, test_size=0.1, random_state=random_seed)

    LOGGER.info("Training regression model")
    regressor = VertexRegressor(model_dir=tb_logger.log_dir, num_epochs=20, batch_size=32)
    regressor.train(X_train, y_train, X_val, y_val)

    LOGGER.info("Regression model trained. Proceeding to metric learning.")

    LOGGER.info("Preparing datasets for training and validation")
    train_data = SPDTimesliceTracksDataset(
        n_samples=train_samples,
        detector_eff=detector_efficiency,
        hits_normalizer=hits_normalizer,
        mode=DatasetMode.train,
        regressor=regressor
    )
    test_data = SPDTimesliceTracksDataset(
        n_samples=test_samples,
        detector_eff=detector_efficiency,
        hits_normalizer=hits_normalizer,
        mode=DatasetMode.val,
        regressor=regressor
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=time_slice_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=time_slice_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )

    LOGGER.info('Creating model for training')
    tracks_embedder = TripletTracksEmbedder(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        triplet_margin=triplet_margin,
        type_of_triplets=type_of_triplets,
        distance=distance,
        metrics=metrics,
        clustering_algorithm=clustering_algorithm(),
    )
    LOGGER.info(tracks_embedder)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        deterministic=True,
        accelerator="auto",
        logger=tb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(
        model=tracks_embedder,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
    )


def main(argv):
    del argv
    gin.parse_config(open(FLAGS.config))
    LOGGER.setLevel(FLAGS.log)
    experiment()
    LOGGER.info("End of training")


if __name__ == "__main__":
    app.run(main)