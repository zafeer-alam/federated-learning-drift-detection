from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass
class ModelSnapshot:
    """
    Stores a snapshot of model parameters with associated performance metadata
    and context identifier.
    """
    name: str
    parameters: Dict[str, torch.Tensor]
    round_created: int
    accuracy: float = 0.0
    context_hash: Optional[str] = None


class ModelPool:
    """
    Bounded retention pool of model snapshots.
    Used under recurring or shifting drift contexts to store, recall,
    and reuse previously mastered model weights without catastrophic forgetting.
    """
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.snapshots: Dict[str, ModelSnapshot] = {}

    def add(self, snapshot: ModelSnapshot) -> None:
        """
        Adds or updates a snapshot in the pool.
        If pool is full, evicts the snapshot with the lowest validation accuracy.
        """
        if snapshot.name in self.snapshots:
            self.snapshots[snapshot.name] = snapshot
            return

        if len(self.snapshots) >= self.max_size:
            # Evict snapshot with the lowest accuracy
            worst_key = min(self.snapshots.keys(), key=lambda k: self.snapshots[k].accuracy)
            del self.snapshots[worst_key]

        self.snapshots[snapshot.name] = snapshot

    def get_best(self, exclude_name: Optional[str] = None) -> Optional[ModelSnapshot]:
        """Returns the snapshot with the highest accuracy in the pool."""
        candidates = [s for k, s in self.snapshots.items() if k != exclude_name]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.accuracy)

    def get_by_context(self, context_hash: str) -> Optional[ModelSnapshot]:
        """Finds a snapshot matching a specific context identifier."""
        for s in self.snapshots.values():
            if s.context_hash == context_hash:
                return s
        return None

    def get_all(self) -> List[ModelSnapshot]:
        """Returns all snapshots in the pool."""
        return list(self.snapshots.values())

    def size(self) -> int:
        """Returns current number of snapshots in pool."""
        return len(self.snapshots)

    def clear(self) -> None:
        """Empties the model pool."""
        self.snapshots.clear()
