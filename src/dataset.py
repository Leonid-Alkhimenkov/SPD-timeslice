import numpy as np
import torch
from enum import IntEnum
from typing import Optional, Callable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .data_generation import SPDEventGenerator
from .regression import VertexRegressor


class DatasetMode(IntEnum):
    train = 0
    val = 1
    test = 2


def time_slice_collator(batch):
    tracks, vertices, event_ids = zip(*batch)
    tracks = [track.clone().detach().float() for track_list in tracks for track in track_list]

    padded_tracks = pad_sequence(tracks, batch_first=True)  # (batch_size, max_num_hits, 3)
    vertices = torch.cat(vertices, dim=0)  # (batch_size, 3)
    vertices = vertices.unsqueeze(1)  # (batch_size, 1, 3)

    combined_input = torch.cat([padded_tracks, vertices], dim=1)  # (batch_size, max_num_hits + 1, 3)

    event_ids = torch.cat(event_ids, dim=0)  # (batch_size,)
    return combined_input, event_ids

    
class SPDTimesliceTracksDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 100,
        detector_eff: float = 0.98,
        mean_events_timeslice: int = 30,
        fixed_num_events: bool = False,
        hits_normalizer: Optional[Callable] = None,
        mode: DatasetMode = DatasetMode.train,
        regressor: Optional[VertexRegressor] = None
    ):
        self.spd_gen = SPDEventGenerator(
            mean_events_timeslice=mean_events_timeslice,
            fixed_num_events=fixed_num_events,
            detector_eff=detector_eff,
            add_fakes=False
        )
        self._n_samples = n_samples
        self.hits_normalizer = hits_normalizer
        self.regressor = regressor
        self._initial_seed = np.random.get_state()[1][mode]

    def __len__(self) -> int:
        return self._n_samples

    # TODO
    def __getitem__(self, idx: int):
        np.random.seed(self._initial_seed + idx)
        time_slice = self.spd_gen.generate_time_slice()

        hits = time_slice.hits
        event_ids = time_slice.event_ids
        vertices = time_slice.vertices
        track_ids = time_slice.track_ids

        unique_track_ids = np.unique(track_ids)

        predicted_vertices = np.zeros((len(unique_track_ids), 3), dtype=np.float32)
        for_predicted_event_ids = np.zeros(len(unique_track_ids), dtype=np.int64)

        tracks = []

        for i, track_id in enumerate(unique_track_ids):
            mask = track_ids == track_id
            track_hits_for_vertex = hits[mask]

            predicted_vertex = self.regressor.predict(track_hits_for_vertex)
            predicted_vertices[i] = predicted_vertex

            for_predicted_event_ids[i] = event_ids[mask][0]

            tracks.append(torch.tensor(track_hits_for_vertex, dtype=torch.float32))
            # print(f"Track {i}: {track_hits_for_vertex.shape}")

        predicted_vertices = torch.tensor(predicted_vertices, dtype=torch.float32)
        for_predicted_event_ids = torch.tensor(for_predicted_event_ids, dtype=torch.long)

        # print(f"Number of tracks: {len(tracks)}")
        # print(f"First track shape: {tracks[0].shape if len(tracks) > 0 else 'No tracks'}")
        return tracks, predicted_vertices, for_predicted_event_ids