from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple
import torch

from src.models.model_registry import ModelPool, ModelSnapshot


class BaseModelSelector(ABC):
    """
    Abstract interface for model selection policies in personalized FL.
    """
    @abstractmethod
    def select(
        self,
        client_id: int,
        round_idx: int,
        global_params: Dict[str, torch.Tensor],
        local_params: Optional[Dict[str, torch.Tensor]],
        pool: Optional[ModelPool] = None,
        detector_alarm: bool = False,
        eval_fn: Optional[Callable[[Dict[str, torch.Tensor]], float]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        """
        Determines which set of parameters the client should adopt.
        Returns:
            - selected_parameters: Dict[str, torch.Tensor]
            - decision_reason: str
        """
        pass


class FixedModelSelector(BaseModelSelector):
    """
    Baseline fixed policy (standard FedAvg / naive personalization).
    Always selects the incoming global model parameters regardless of drift status.
    """
    def select(
        self,
        client_id: int,
        round_idx: int,
        global_params: Dict[str, torch.Tensor],
        local_params: Optional[Dict[str, torch.Tensor]],
        pool: Optional[ModelPool] = None,
        detector_alarm: bool = False,
        eval_fn: Optional[Callable[[Dict[str, torch.Tensor]], float]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        return global_params, "fixed_global"


class DriftAwareModelSelector(BaseModelSelector):
    """
    Drift-aware model selection policy.
    Under stationary conditions: adopts the global model.
    Upon detector alarm: evaluates candidate models (global model, local model,
    and any retained pool snapshots) on recent local validation data and selects
    the candidate with the highest accuracy.
    """
    def __init__(self, fallback_to_local: bool = True):
        self.fallback_to_local = fallback_to_local

    def select(
        self,
        client_id: int,
        round_idx: int,
        global_params: Dict[str, torch.Tensor],
        local_params: Optional[Dict[str, torch.Tensor]],
        pool: Optional[ModelPool] = None,
        detector_alarm: bool = False,
        eval_fn: Optional[Callable[[Dict[str, torch.Tensor]], float]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        if not detector_alarm or eval_fn is None:
            return global_params, "stationary_global"

        # Construct candidate set
        candidates: Dict[str, Dict[str, torch.Tensor]] = {"global": global_params}
        if local_params is not None and self.fallback_to_local:
            candidates["local"] = local_params

        if pool is not None:
            for s in pool.get_all():
                candidates[f"pool_{s.name}"] = s.parameters

        # Evaluate all candidate models
        best_name = "global"
        best_score = float("-inf")

        for name, params in candidates.items():
            score = eval_fn(params)
            if score > best_score:
                best_score = score
                best_name = name

        return candidates[best_name], f"drift_adapted_{best_name}"


def get_model_selector(name: str = "fixed", **kwargs) -> BaseModelSelector:
    """Factory function for instantiating model selection policies."""
    clean_name = name.lower().strip()
    if clean_name == "fixed":
        return FixedModelSelector()
    elif clean_name in ["drift_aware", "driftaware", "adaptive"]:
        return DriftAwareModelSelector(**kwargs)
    else:
        raise ValueError(f"Unknown selector '{name}'. Supported: 'fixed', 'drift_aware'")
