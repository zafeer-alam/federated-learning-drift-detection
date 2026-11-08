"""
Communication schedulers for federated learning.

Implements three scheduling policies that determine *when* a client should
communicate (send/receive model updates) with the server:

* **FixedScheduler** — standard periodic communication every τ rounds.
* **DriftTriggeredScheduler** — communicates only when a drift detector fires.
* **HybridScheduler** — combines periodic + drift-triggered communication.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseCommunicationScheduler(ABC):
    """Abstract base class for communication scheduling policies.

    Every concrete scheduler tracks the total number of communications
    performed and maintains a structured log of each event.
    """

    def __init__(self) -> None:
        self.total_communications: int = 0
        self.communication_log: List[Dict[str, Any]] = []

    @abstractmethod
    def should_communicate(
        self,
        client_id: int,
        round_idx: int,
        detector_alarm: bool = False,
    ) -> bool:
        """Decide whether a client should communicate this round.

        Args:
            client_id: Identifier of the FL client.
            round_idx: Current training round index (0-based).
            detector_alarm: ``True`` when the drift detector has fired.

        Returns:
            ``True`` if the client should synchronise with the server.
        """
        pass

    def record_communication(
        self,
        client_id: int,
        round_idx: int,
        reason: str,
    ) -> None:
        """Log a communication event.

        Args:
            client_id: Identifier of the FL client.
            round_idx: Round in which communication took place.
            reason: Human-readable description (e.g. ``"periodic"``).
        """
        self.total_communications += 1
        self.communication_log.append(
            {"client_id": client_id, "round": round_idx, "reason": reason}
        )


class FixedScheduler(BaseCommunicationScheduler):
    """Communicates every *τ* rounds (standard periodic scheduling).

    Args:
        tau: Communication period.  ``tau=1`` means every round (default).
    """

    def __init__(self, tau: int = 1) -> None:
        super().__init__()
        if tau < 1:
            raise ValueError(f"tau must be >= 1, got {tau}")
        self.tau: int = tau

    def should_communicate(
        self,
        client_id: int,
        round_idx: int,
        detector_alarm: bool = False,
    ) -> bool:
        """Return ``True`` when ``round_idx`` is a multiple of *τ*."""
        return round_idx % self.tau == 0


class DriftTriggeredScheduler(BaseCommunicationScheduler):
    """Communicates **only** when the drift detector signals an alarm.

    This scheduler is fully reactive: it never initiates communication
    on its own schedule.
    """

    def should_communicate(
        self,
        client_id: int,
        round_idx: int,
        detector_alarm: bool = False,
    ) -> bool:
        """Return ``True`` when ``detector_alarm`` is set."""
        return detector_alarm


class HybridScheduler(BaseCommunicationScheduler):
    """Combines periodic scheduling with drift-triggered overrides.

    Communication occurs if *either* of the following conditions holds:

    1. The round is a periodic communication round (``round_idx % tau == 0``).
    2. ``allow_drift_override`` is enabled *and* the drift detector fires.

    Args:
        tau: Communication period for the periodic component (default 5).
        allow_drift_override: If ``True`` (default), a drift alarm triggers
            immediate communication regardless of the periodic schedule.
    """

    def __init__(
        self,
        tau: int = 5,
        allow_drift_override: bool = True,
    ) -> None:
        super().__init__()
        if tau < 1:
            raise ValueError(f"tau must be >= 1, got {tau}")
        self.tau: int = tau
        self.allow_drift_override: bool = allow_drift_override

    def should_communicate(
        self,
        client_id: int,
        round_idx: int,
        detector_alarm: bool = False,
    ) -> bool:
        """Return ``True`` on periodic rounds or on drift alarm (if enabled)."""
        periodic = round_idx % self.tau == 0
        drift_override = self.allow_drift_override and detector_alarm
        return periodic or drift_override


def get_scheduler(name: str = "fixed", **kwargs: Any) -> BaseCommunicationScheduler:
    """Factory function for instantiating communication schedulers.

    Args:
        name: Scheduler type.  Supported values: ``'fixed'``,
            ``'drift_triggered'``, ``'hybrid'``.
        **kwargs: Forwarded to the scheduler constructor.

    Returns:
        An instance of the requested scheduler.

    Raises:
        ValueError: If *name* is not one of the supported scheduler types.
    """
    name_clean = name.lower().strip()
    if name_clean == "fixed":
        return FixedScheduler(**kwargs)
    elif name_clean == "drift_triggered":
        return DriftTriggeredScheduler(**kwargs)
    elif name_clean == "hybrid":
        return HybridScheduler(**kwargs)
    else:
        raise ValueError(
            f"Unknown scheduler: '{name}'. "
            "Supported: 'fixed', 'drift_triggered', 'hybrid'"
        )
