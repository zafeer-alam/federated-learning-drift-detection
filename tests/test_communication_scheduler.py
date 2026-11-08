"""Unit tests for the communication scheduler module."""
import pytest

from src.communication.scheduler import (
    BaseCommunicationScheduler,
    DriftTriggeredScheduler,
    FixedScheduler,
    HybridScheduler,
    get_scheduler,
)


# ---------------------------------------------------------------------------
# FixedScheduler
# ---------------------------------------------------------------------------

class TestFixedScheduler:
    """Tests for FixedScheduler."""

    def test_tau_1_always_communicates(self) -> None:
        """With tau=1 every round should trigger communication."""
        sched = FixedScheduler(tau=1)
        for r in range(20):
            assert sched.should_communicate(client_id=0, round_idx=r) is True

    def test_tau_3_communicates_every_third_round(self) -> None:
        """With tau=3 communication happens at rounds 0, 3, 6, 9, …"""
        sched = FixedScheduler(tau=3)
        for r in range(15):
            expected = r % 3 == 0
            assert sched.should_communicate(client_id=0, round_idx=r) is expected, (
                f"round {r}: expected {expected}"
            )

    def test_tau_invalid_raises(self) -> None:
        """tau < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            FixedScheduler(tau=0)
        with pytest.raises(ValueError):
            FixedScheduler(tau=-1)

    def test_detector_alarm_ignored(self) -> None:
        """FixedScheduler should not care about detector_alarm."""
        sched = FixedScheduler(tau=2)
        # Round 1 is not periodic → should be False even with alarm
        assert sched.should_communicate(client_id=0, round_idx=1, detector_alarm=True) is False
        # Round 2 is periodic → True regardless
        assert sched.should_communicate(client_id=0, round_idx=2, detector_alarm=False) is True


# ---------------------------------------------------------------------------
# DriftTriggeredScheduler
# ---------------------------------------------------------------------------

class TestDriftTriggeredScheduler:
    """Tests for DriftTriggeredScheduler."""

    def test_no_alarm_no_communication(self) -> None:
        """Without an alarm the scheduler should never communicate."""
        sched = DriftTriggeredScheduler()
        for r in range(20):
            assert sched.should_communicate(client_id=0, round_idx=r) is False

    def test_alarm_triggers_communication(self) -> None:
        """When detector_alarm is True, communication should happen."""
        sched = DriftTriggeredScheduler()
        assert sched.should_communicate(client_id=0, round_idx=5, detector_alarm=True) is True

    def test_mixed_alarms(self) -> None:
        """Only rounds with alarm=True should communicate."""
        sched = DriftTriggeredScheduler()
        alarms = [False, False, True, False, True, False]
        for r, alarm in enumerate(alarms):
            assert sched.should_communicate(client_id=0, round_idx=r, detector_alarm=alarm) is alarm


# ---------------------------------------------------------------------------
# HybridScheduler
# ---------------------------------------------------------------------------

class TestHybridScheduler:
    """Tests for HybridScheduler."""

    def test_periodic_component(self) -> None:
        """On periodic rounds (no alarm) communication should happen."""
        sched = HybridScheduler(tau=5)
        assert sched.should_communicate(client_id=0, round_idx=0) is True
        assert sched.should_communicate(client_id=0, round_idx=5) is True
        assert sched.should_communicate(client_id=0, round_idx=10) is True

    def test_non_periodic_no_alarm(self) -> None:
        """Non-periodic rounds without alarm should not communicate."""
        sched = HybridScheduler(tau=5)
        for r in [1, 2, 3, 4, 6, 7]:
            assert sched.should_communicate(client_id=0, round_idx=r) is False

    def test_drift_override(self) -> None:
        """Alarm on a non-periodic round should still trigger communication."""
        sched = HybridScheduler(tau=5, allow_drift_override=True)
        assert sched.should_communicate(client_id=0, round_idx=3, detector_alarm=True) is True

    def test_drift_override_disabled(self) -> None:
        """When allow_drift_override=False, alarm alone is not enough."""
        sched = HybridScheduler(tau=5, allow_drift_override=False)
        assert sched.should_communicate(client_id=0, round_idx=3, detector_alarm=True) is False
        # Periodic round still works
        assert sched.should_communicate(client_id=0, round_idx=5, detector_alarm=False) is True

    def test_both_conditions_met(self) -> None:
        """Alarm on a periodic round should still return True (OR logic)."""
        sched = HybridScheduler(tau=5)
        assert sched.should_communicate(client_id=0, round_idx=10, detector_alarm=True) is True

    def test_tau_invalid_raises(self) -> None:
        """tau < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            HybridScheduler(tau=0)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

class TestGetScheduler:
    """Tests for the get_scheduler factory."""

    def test_fixed(self) -> None:
        sched = get_scheduler("fixed", tau=2)
        assert isinstance(sched, FixedScheduler)
        assert sched.tau == 2

    def test_drift_triggered(self) -> None:
        sched = get_scheduler("drift_triggered")
        assert isinstance(sched, DriftTriggeredScheduler)

    def test_hybrid(self) -> None:
        sched = get_scheduler("hybrid", tau=10)
        assert isinstance(sched, HybridScheduler)
        assert sched.tau == 10

    def test_case_insensitive(self) -> None:
        sched = get_scheduler("  Fixed  ")
        assert isinstance(sched, FixedScheduler)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            get_scheduler("unknown_scheduler")


# ---------------------------------------------------------------------------
# Communication logging
# ---------------------------------------------------------------------------

class TestCommunicationLogging:
    """Tests for record_communication on the base interface."""

    def test_logging_increments_counter(self) -> None:
        sched = FixedScheduler(tau=1)
        assert sched.total_communications == 0
        sched.record_communication(client_id=0, round_idx=0, reason="periodic")
        assert sched.total_communications == 1
        sched.record_communication(client_id=1, round_idx=1, reason="periodic")
        assert sched.total_communications == 2

    def test_logging_records_entries(self) -> None:
        sched = DriftTriggeredScheduler()
        sched.record_communication(client_id=3, round_idx=7, reason="drift_alarm")
        assert len(sched.communication_log) == 1
        entry = sched.communication_log[0]
        assert entry["client_id"] == 3
        assert entry["round"] == 7
        assert entry["reason"] == "drift_alarm"

    def test_logging_multiple_entries(self) -> None:
        sched = HybridScheduler(tau=5)
        for i in range(5):
            sched.record_communication(client_id=i, round_idx=i * 5, reason="periodic")
        assert sched.total_communications == 5
        assert len(sched.communication_log) == 5
        assert sched.communication_log[2]["client_id"] == 2


if __name__ == "__main__":
    pytest.main([__file__])
