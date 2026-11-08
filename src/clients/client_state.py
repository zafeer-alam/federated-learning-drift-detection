from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch


@dataclass
class ClientState:
    """
    Encapsulates internal client state, memory, and telemetry.
    """
    client_id: int
    current_round: int = 0
    num_train_samples: int = 0
    num_test_samples: int = 0
    cumulative_bytes_sent: int = 0
    cumulative_bytes_received: int = 0
    total_local_epochs: int = 0
    training_time_seconds: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)
    local_parameters: Optional[Dict[str, torch.Tensor]] = None
    personalized_parameters: Optional[Dict[str, torch.Tensor]] = None

    def record_communication(self, bytes_sent: int = 0, bytes_received: int = 0) -> None:
        """Accumulates network bandwidth metrics."""
        self.cumulative_bytes_sent += bytes_sent
        self.cumulative_bytes_received += bytes_received

    def record_round_metrics(self, round_idx: int, metrics: Dict[str, Any]) -> None:
        """Stores per-round historical telemetry."""
        record = {"round": round_idx, **metrics}
        self.history.append(record)
