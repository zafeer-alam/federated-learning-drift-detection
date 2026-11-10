import pytest
import torch

from src.models.model_registry import ModelPool, ModelSnapshot
from src.selection.model_selector import (
    BaseModelSelector,
    DriftAwareModelSelector,
    FixedModelSelector,
    get_model_selector,
)


def _make_dummy_params(val: float):
    return {"weight": torch.tensor([val])}


def test_model_pool_add_and_get():
    pool = ModelPool(max_size=3)
    s1 = ModelSnapshot(name="m1", parameters=_make_dummy_params(1.0), round_created=1, accuracy=0.8)
    s2 = ModelSnapshot(name="m2", parameters=_make_dummy_params(2.0), round_created=2, accuracy=0.9)

    pool.add(s1)
    pool.add(s2)
    assert pool.size() == 2

    best = pool.get_best()
    assert best is not None
    assert best.name == "m2"
    assert best.accuracy == 0.9


def test_model_pool_eviction():
    pool = ModelPool(max_size=2)
    s1 = ModelSnapshot(name="m1", parameters=_make_dummy_params(1.0), round_created=1, accuracy=0.7)
    s2 = ModelSnapshot(name="m2", parameters=_make_dummy_params(2.0), round_created=2, accuracy=0.9)
    s3 = ModelSnapshot(name="m3", parameters=_make_dummy_params(3.0), round_created=3, accuracy=0.8)

    pool.add(s1)
    pool.add(s2)
    pool.add(s3)  # Should evict m1 (lowest accuracy 0.7)

    assert pool.size() == 2
    names = [s.name for s in pool.get_all()]
    assert "m1" not in names
    assert "m2" in names
    assert "m3" in names


def test_fixed_model_selector():
    selector = FixedModelSelector()
    global_p = _make_dummy_params(1.0)
    local_p = _make_dummy_params(2.0)

    selected, reason = selector.select(
        client_id=0,
        round_idx=1,
        global_params=global_p,
        local_params=local_p,
        detector_alarm=True,  # Even with alarm, fixed policy always returns global
    )
    assert selected == global_p
    assert reason == "fixed_global"


def test_drift_aware_model_selector_no_alarm():
    selector = DriftAwareModelSelector()
    global_p = _make_dummy_params(1.0)
    local_p = _make_dummy_params(2.0)

    selected, reason = selector.select(
        client_id=0,
        round_idx=1,
        global_params=global_p,
        local_params=local_p,
        detector_alarm=False,
    )
    assert selected == global_p
    assert "stationary" in reason


def test_drift_aware_model_selector_with_alarm():
    selector = DriftAwareModelSelector()
    global_p = _make_dummy_params(1.0)
    local_p = _make_dummy_params(2.0)
    pool_p = _make_dummy_params(3.0)

    pool = ModelPool(max_size=3)
    pool.add(ModelSnapshot(name="retained_v1", parameters=pool_p, round_created=0, accuracy=0.95))

    # Evaluation function where retained pool parameters achieve highest score
    def mock_eval(params):
        val = params["weight"].item()
        if val == 3.0:
            return 0.95
        elif val == 2.0:
            return 0.70
        else:
            return 0.40

    selected, reason = selector.select(
        client_id=0,
        round_idx=5,
        global_params=global_p,
        local_params=local_p,
        pool=pool,
        detector_alarm=True,
        eval_fn=mock_eval,
    )

    assert selected == pool_p
    assert "retained_v1" in reason


def test_get_model_selector_factory():
    fixed = get_model_selector("fixed")
    assert isinstance(fixed, FixedModelSelector)

    adaptive = get_model_selector("drift_aware")
    assert isinstance(adaptive, DriftAwareModelSelector)

    with pytest.raises(ValueError):
        get_model_selector("unsupported_selector")
