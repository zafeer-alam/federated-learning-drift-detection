import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.drift.drift_simulator import (
    DriftEvent,
    DriftSimulator,
    DriftType,
    create_abrupt_drift_schedule,
    create_gradual_drift_schedule,
    create_recurring_drift_schedule,
    create_staggered_drift_schedule,
)


def test_abrupt_drift_schedule():
    sim = DriftSimulator(num_classes=10, seed=42)
    events = create_abrupt_drift_schedule(client_ids=[0, 1], drift_round=10)
    for e in events:
        sim.register_schedule(e)

    # Before drift onset: round 9
    p0_pre, _ = sim.get_drift_probability(client_id=0, round_idx=9)
    assert p0_pre == 0.0

    # At and after onset: round 10 and 15
    p0_at, _ = sim.get_drift_probability(client_id=0, round_idx=10)
    p0_post, _ = sim.get_drift_probability(client_id=0, round_idx=15)
    assert p0_at == 1.0
    assert p0_post == 1.0

    # Unregistered client: client 2
    p2, _ = sim.get_drift_probability(client_id=2, round_idx=15)
    assert p2 == 0.0


def test_gradual_drift_probability_ramp():
    sim = DriftSimulator(num_classes=10, seed=42)
    events = create_gradual_drift_schedule(client_ids=[0], start_round=10, end_round=20)
    sim.register_schedule(events[0])

    p_before, _ = sim.get_drift_probability(0, 9)
    p_start, _ = sim.get_drift_probability(0, 10)
    p_mid, _ = sim.get_drift_probability(0, 15)
    p_end, _ = sim.get_drift_probability(0, 20)
    p_after, _ = sim.get_drift_probability(0, 25)

    assert p_before == 0.0
    assert p_start == 0.0
    assert abs(p_mid - 0.5) < 1e-4
    assert p_end == 1.0
    assert p_after == 1.0


def test_recurring_drift_intervals():
    sim = DriftSimulator(num_classes=10, seed=42)
    # Drift active during [10, 20] and [30, 40]
    events = create_recurring_drift_schedule(client_ids=[0], recurrence_intervals=[(10, 20), (30, 40)])
    for e in events:
        sim.register_schedule(e)

    p_baseline1, _ = sim.get_drift_probability(0, 5)
    p_drift1, _ = sim.get_drift_probability(0, 15)
    p_baseline2, _ = sim.get_drift_probability(0, 25)
    p_drift2, _ = sim.get_drift_probability(0, 35)
    p_baseline3, _ = sim.get_drift_probability(0, 45)

    assert p_baseline1 == 0.0
    assert p_drift1 == 1.0
    assert p_baseline2 == 0.0
    assert p_drift2 == 1.0
    assert p_baseline3 == 0.0


def test_staggered_drift_onsets():
    sim = DriftSimulator(num_classes=10, seed=42)
    onset_map = {0: 5, 1: 15}
    events = create_staggered_drift_schedule(onset_map)
    for e in events:
        sim.register_schedule(e)

    p0, _ = sim.get_drift_probability(0, 10)
    p1, _ = sim.get_drift_probability(1, 10)

    assert p0 == 1.0, "Client 0 should be drifted at round 10 (onset=5)"
    assert p1 == 0.0, "Client 1 should NOT be drifted at round 10 (onset=15)"


def test_label_permutation_transformation():
    sim = DriftSimulator(num_classes=10, seed=42)
    events = create_abrupt_drift_schedule(client_ids=[0], drift_round=5, transformation="label_permute")
    sim.register_schedule(events[0])

    data = torch.randn(10, 1, 28, 28)
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Pre-drift: round 4
    _, t_pre, active_pre = sim.apply_drift_to_batch(0, 4, data, targets)
    assert not active_pre
    assert torch.equal(t_pre, targets)

    # Post-drift: round 6
    _, t_post, active_post = sim.apply_drift_to_batch(0, 6, data, targets)
    assert active_post
    # Labels should be shifted by 5 (num_classes // 2): (0->5, 1->6, ..., 5->0)
    expected_post = (targets + 5) % 10
    assert torch.equal(t_post, expected_post)


def test_image_rotation_transformation():
    sim = DriftSimulator(num_classes=10, seed=42)
    events = create_abrupt_drift_schedule(client_ids=[0], drift_round=5, transformation="rotate_90")
    sim.register_schedule(events[0])

    data = torch.zeros(2, 1, 4, 4)
    data[:, :, 0, 0] = 1.0 # top-left pixel = 1.0
    targets = torch.tensor([0, 1])

    d_post, _, active = sim.apply_drift_to_batch(0, 6, data, targets)
    assert active
    # After 90-degree counter-clockwise rotation, pixel moves from [0, 0] to [3, 0]
    assert d_post[0, 0, 3, 0].item() == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
