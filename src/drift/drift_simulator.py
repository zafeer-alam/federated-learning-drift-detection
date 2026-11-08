from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DriftType(str, Enum):
    NONE = "none"
    ABRUPT = "abrupt"
    GRADUAL = "gradual"
    RECURRING = "recurring"
    STAGGERED = "staggered"


@dataclass
class DriftEvent:
    """
    Specifies a ground-truth concept drift schedule for a specific client.
    """
    client_id: int
    drift_type: DriftType
    start_round: int
    end_round: Optional[int] = None
    transformation: str = "label_permute" # "label_permute", "rotate_90", "invert"
    param: Any = None
    description: str = ""


class DriftedDataset(Dataset):
    """
    Wraps an underlying PyTorch Dataset and applies dynamic drift transformations.
    """
    def __init__(self, base_dataset: Dataset, transform_fn: Callable[[torch.Tensor, int], Tuple[torch.Tensor, int]]):
        self.base_dataset = base_dataset
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        return self.transform_fn(x, y)


class DriftSimulator:
    """
    Deterministic drift injection simulator.
    Manages client drift schedules, applies transformations to batches or datasets,
    and logs the ground truth drift timeline for rigorous detector evaluation.
    """
    def __init__(self, num_classes: int = 10, seed: int = 42):
        self.num_classes = num_classes
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.schedules: Dict[int, List[DriftEvent]] = {}
        self.ground_truth_log: List[Dict[str, Any]] = []

    def register_schedule(self, event: DriftEvent) -> None:
        """Registers a drift event for a specific client."""
        if event.client_id not in self.schedules:
            self.schedules[event.client_id] = []
        self.schedules[event.client_id].append(event)

    def get_drift_probability(self, client_id: int, round_idx: int) -> Tuple[float, Optional[DriftEvent]]:
        """
        Determines whether client_id experiences drift at round_idx, and returns
        the mixture probability p in [0, 1] of applying the drift transformation.
        """
        if client_id not in self.schedules:
            return 0.0, None

        for event in self.schedules[client_id]:
            if event.drift_type == DriftType.ABRUPT or event.drift_type == DriftType.STAGGERED:
                if round_idx >= event.start_round:
                    if event.end_round is None or round_idx <= event.end_round:
                        return 1.0, event
            elif event.drift_type == DriftType.GRADUAL:
                if event.end_round is not None and event.start_round <= round_idx <= event.end_round:
                    span = max(event.end_round - event.start_round, 1)
                    p = float(round_idx - event.start_round) / float(span)
                    return min(max(p, 0.0), 1.0), event
                elif event.end_round is not None and round_idx > event.end_round:
                    return 1.0, event
            elif event.drift_type == DriftType.RECURRING:
                if event.end_round is not None and event.start_round <= round_idx <= event.end_round:
                    return 1.0, event

        return 0.0, None

    def transform_sample(
        self,
        x: torch.Tensor,
        y: int,
        transformation: str = "label_permute",
    ) -> Tuple[torch.Tensor, int]:
        """Applies transformation to an individual data point."""
        if transformation == "label_permute":
            # Deterministic pairwise label swap: (0<->5, 1<->6, 2<->7, 3<->8, 4<->9)
            new_y = (y + self.num_classes // 2) % self.num_classes
            return x, new_y
        elif transformation == "rotate_90":
            # 90-degree spatial image rotation
            new_x = torch.rot90(x, 1, [-2, -1])
            return new_x, y
        elif transformation == "invert":
            # Invert normalized pixels
            new_x = -x
            return new_x, y
        else:
            return x, y

    def apply_drift_to_batch(
        self,
        client_id: int,
        round_idx: int,
        data: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Applies scheduled drift to a mini-batch of tensors.
        Returns (transformed_data, transformed_target, is_drift_active).
        """
        prob, event = self.get_drift_probability(client_id, round_idx)
        is_active = prob > 0.0

        # Log ground truth record
        self.ground_truth_log.append({
            "client_id": client_id,
            "round": round_idx,
            "is_drift_active": is_active,
            "drift_probability": prob,
            "drift_type": event.drift_type.value if event else "none",
            "transformation": event.transformation if event else "none",
        })

        if not is_active:
            return data, target, False

        transformed_data = data.clone()
        transformed_target = target.clone()

        for i in range(len(target)):
            # For gradual drift, sample Bernoulli(prob)
            if prob >= 1.0 or np.random.rand() < prob:
                t_x, t_y = self.transform_sample(
                    data[i], int(target[i].item()), transformation=event.transformation
                )
                transformed_data[i] = t_x
                transformed_target[i] = t_y

        return transformed_data, transformed_target, True

    def wrap_loader(
        self,
        client_id: int,
        round_idx: int,
        loader: DataLoader,
    ) -> DataLoader:
        """
        Wraps a client's DataLoader with dynamic drift transformation for the current round.
        """
        prob, event = self.get_drift_probability(client_id, round_idx)
        if prob <= 0.0 or event is None:
            return loader

        def transform_fn(x, y):
            if prob >= 1.0 or np.random.rand() < prob:
                return self.transform_sample(x, int(y), transformation=event.transformation)
            return x, y

        drifted_dataset = DriftedDataset(loader.dataset, transform_fn)
        return DataLoader(
            drifted_dataset,
            batch_size=loader.batch_size,
            shuffle=True if loader.sampler is not None else False,
        )


def create_abrupt_drift_schedule(
    client_ids: List[int],
    drift_round: int = 20,
    transformation: str = "label_permute",
) -> List[DriftEvent]:
    """Creates abrupt drift schedule at drift_round for specified clients."""
    return [
        DriftEvent(
            client_id=cid,
            drift_type=DriftType.ABRUPT,
            start_round=drift_round,
            end_round=None,
            transformation=transformation,
            description=f"Abrupt drift onset at round {drift_round}",
        )
        for cid in client_ids
    ]


def create_gradual_drift_schedule(
    client_ids: List[int],
    start_round: int = 15,
    end_round: int = 25,
    transformation: str = "label_permute",
) -> List[DriftEvent]:
    """Creates gradual sigmoid-like transition schedule over [start_round, end_round]."""
    return [
        DriftEvent(
            client_id=cid,
            drift_type=DriftType.GRADUAL,
            start_round=start_round,
            end_round=end_round,
            transformation=transformation,
            description=f"Gradual drift transition over rounds {start_round}-{end_round}",
        )
        for cid in client_ids
    ]


def create_recurring_drift_schedule(
    client_ids: List[int],
    recurrence_intervals: List[Tuple[int, int]], # e.g. [(15, 25), (35, 45)]
    transformation: str = "label_permute",
) -> List[DriftEvent]:
    """Creates recurring drift schedule where drifted context B is active during specified intervals."""
    events = []
    for cid in client_ids:
        for start_r, end_r in recurrence_intervals:
            events.append(
                DriftEvent(
                    client_id=cid,
                    drift_type=DriftType.RECURRING,
                    start_round=start_r,
                    end_round=end_r,
                    transformation=transformation,
                    description=f"Recurring context active rounds {start_r}-{end_r}",
                )
            )
    return events


def create_staggered_drift_schedule(
    client_onset_map: Dict[int, int], # e.g. {0: 10, 1: 15, 2: 20, 3: 25}
    transformation: str = "label_permute",
) -> List[DriftEvent]:
    """Creates staggered drift schedule where each client drifts at its own onset round."""
    events = []
    for cid, onset_r in client_onset_map.items():
        events.append(
            DriftEvent(
                client_id=cid,
                drift_type=DriftType.STAGGERED,
                start_round=onset_r,
                end_round=None,
                transformation=transformation,
                description=f"Staggered drift onset for client {cid} at round {onset_r}",
            )
        )
    return events
