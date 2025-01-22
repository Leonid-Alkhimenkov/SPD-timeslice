import gin
import importlib

import torch
from torch import nn
from enum import Enum
from umap import UMAP

import pytorch_lightning as pl
from pytorch_metric_learning import (
    distances,
    reducers,
    losses,
    miners,
)

import numpy as np
from sklearn.cluster import KMeans

from collections import defaultdict
from typing import Optional, Type, Any, Dict

from .metrics import *
from .clustering import *
from .visualization import draw_embeddings


@gin.constants_from_enum
class TripletType(str, Enum):
    all = "all"
    semihard = "semihard"
    hard = "hard"
    easy = "easy"


@gin.constants_from_enum
class DistanceType(str, Enum):
    cosine_similarity = "CosineSimilarity"
    euclidean_distance = "LpDistance"
    dot_product = "DotProductSimilarity"
    snr_distance = "SNRDistance"


class TripletTracksEmbedder(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        triplet_margin: float = 0.2,
        type_of_triplets: TripletType = "semihard",
        distance: DistanceType = DistanceType.cosine_similarity,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        umapper: Optional[UMAP] = None,
        clustering_algorithm: Optional[Any] = None,
        metrics: Optional[list] = None,
    ):
        super().__init__()

        if umapper is None:
            umapper = UMAP()
        self.umapper = umapper

        self.model = model
        self._distance = getattr(distances, distance)()
        self.criterion = losses.TripletMarginLoss(
            margin=triplet_margin,
            distance=self._distance,
            reducer=reducers.ThresholdReducer(low=0)
        )

        if clustering_algorithm is None:
            clustering_algorithm = KMeans(n_clusters=40, random_state=42, n_init=10)

        self.clustering = Clustering(clustering_algorithm)

        self.metrics = metrics
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.triplet_miner = miners.TripletMarginMiner(
            margin=triplet_margin,
            type_of_triplets=type_of_triplets,
            distance=self._distance
        )
        self.validation_step_outputs = []

        self.epoch_metrics = defaultdict(list)

    def forward(self, inputs):
        return self.model(inputs)

    def _forward_batch(self, batch, return_embeddings=False):
        """
        Обрабатывает батч данных, вычисляет эмбеддинги и функцию потерь.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        combined_input, event_ids = batch

        combined_input = combined_input.to(device)
        event_ids = event_ids.to(device)
        self.model = self.model.to(device)

        embeddings = self.model(combined_input)

        indices_tuple = self.triplet_miner(embeddings, event_ids)
        loss = self.criterion(embeddings, event_ids, indices_tuple)

        if return_embeddings:
            return loss, embeddings
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._forward_batch(batch)
        self.log_dict(
            {
                "train_loss": loss,
                "train_triplets": float(self.triplet_miner.num_triplets)
            },
            prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, embeddings = self._forward_batch(batch, return_embeddings=True)
        emb_np = embeddings.cpu().detach().numpy()
        evt_ids_np = batch[1].cpu().numpy()

        evt_ids_np_prep, cluster_preds, cluster_assignments = self.clustering.cluster_and_link(
            emb_np, evt_ids_np
        )

        # unique_labels = np.unique(cluster_assignments)
        # print(unique_labels)
        # if len(unique_labels) < 2:
        #     print("val_warning", "Skipping metrics calculation: only one unique label found.")
        #     return

        for metric in self.metrics:
            if isinstance(metric, (
                F1ScoreMetric,
                PrecisionScoreMetric,
                RecallScoreMetric,
                AccuracyScoreMetric
            )):
                metric.update(evt_ids_np_prep, cluster_preds)
            elif isinstance(metric, BaseScoreMetric):
                metric.update(emb_np, cluster_assignments)

        self.validation_step_outputs.append({
            "embeddings": emb_np,
            "event_ids": evt_ids_np,
        })

        self.log_dict(
            {
                "val_loss": loss,
                "val_triplets": float(self.triplet_miner.num_triplets),
            },
            prog_bar=True
        )

    def on_validation_epoch_end(self):
        sample_idx = 0
        sample_for_visualization = self.validation_step_outputs[sample_idx]

        for metric in self.metrics:
            metric_name = type(metric).__name__
            self.logger.experiment.add_scalar(
                f'{metric_name}',
                metric.compute(),
                self.current_epoch
            )

        umap_embeddings = self.umapper.fit_transform(
            sample_for_visualization["embeddings"]
        )

        plt_obj = draw_embeddings(
            embeddings=umap_embeddings,
            labels=sample_for_visualization["event_ids"],
            sample_idx=sample_idx,
            split_name="validation"
        )

        self.logger.experiment.add_figure(
            "Embeddings visualization", plt_obj.gcf(), self.current_epoch
        )

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer