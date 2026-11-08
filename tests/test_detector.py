import numpy as np
import pytest

from src.drift.detector import (
    ADWINDetector,
    DDMDetector,
    PageHinkleyDetector,
    get_detector,
)


def test_ddm_detector():
    detector = DDMDetector(min_instances=30)
    
    # Stationary low error phase: 95% correct
    np.random.seed(42)
    stationary_stream = [1 if np.random.rand() < 0.05 else 0 for _ in range(60)]
    for val in stationary_stream:
        drift = detector.update(val)
        assert not drift, "DDM should not signal drift during stationary phase"

    # Sudden high error rate phase: 90% error
    drift_signaled = False
    drift_stream = [1 if np.random.rand() < 0.90 else 0 for _ in range(60)]
    for val in drift_stream:
        if detector.update(val):
            drift_signaled = True
            break

    assert drift_signaled, "DDM must signal drift when error rate surges"


def test_adwin_detector():
    detector = ADWINDetector(delta=0.01, max_window_size=100, min_subwindow_size=10)
    
    # Stream from Distribution 1: N(0, 0.1)
    np.random.seed(42)
    s1 = np.random.normal(0.0, 0.1, 40)
    for val in s1:
        drift = detector.update(val)
        assert not drift, "ADWIN should not trigger during stationary distribution"

    # Stream from Distribution 2: N(3.0, 0.1)
    s2 = np.random.normal(3.0, 0.1, 40)
    drift_signaled = False
    for val in s2:
        if detector.update(val):
            drift_signaled = True
            break

    assert drift_signaled, "ADWIN must detect distribution shift in mean"


def test_page_hinkley_detector():
    detector = PageHinkleyDetector(threshold=20.0, delta=0.01)
    
    # Low loss regime: 0.1
    for _ in range(50):
        drift = detector.update(0.1)
        assert not drift

    # Loss jump regime: 2.0
    drift_signaled = False
    for _ in range(30):
        if detector.update(2.0):
            drift_signaled = True
            break

    assert drift_signaled, "Page-Hinkley must detect sharp loss surge"


def test_detector_factory():
    adwin = get_detector("adwin")
    assert isinstance(adwin, ADWINDetector)

    ddm = get_detector("ddm")
    assert isinstance(ddm, DDMDetector)

    ph = get_detector("page_hinkley")
    assert isinstance(ph, PageHinkleyDetector)

    with pytest.raises(ValueError):
        get_detector("unknown_detector")


if __name__ == "__main__":
    pytest.main([__file__])
