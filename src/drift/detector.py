from abc import ABC, abstractmethod
import math
from typing import List, Optional
import numpy as np


class BaseDriftDetector(ABC):
    """
    Abstract base interface for online drift detectors.
    Processes scalar observations sequentially and signals when distribution changes occur.
    """
    def __init__(self):
        self.drift_detected: bool = False
        self.warning_detected: bool = False
        self.total_samples: int = 0

    @abstractmethod
    def update(self, val: float) -> bool:
        """
        Updates detector with a new scalar observation.
        Returns True if concept drift is signaled.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets detector internal statistics."""
        pass


class DDMDetector(BaseDriftDetector):
    """
    Drift Detection Method (DDM) based on Gama et al. (2004) 'Learning with Drift Detection'.
    Monitors a Bernoulli error stream (0 = correct, 1 = classification error).
    Signals warning when p + s >= p_min + 2 * s_min.
    Signals drift when p + s >= p_min + 3 * s_min.
    """
    def __init__(self, min_instances: int = 30, warning_level: float = 2.0, out_control_level: float = 3.0):
        super().__init__()
        self.min_instances = min_instances
        self.warning_level = warning_level
        self.out_control_level = out_control_level
        self.reset()

    def reset(self) -> None:
        self.total_samples = 0
        self.p: float = 1.0 # Current error rate
        self.s: float = 0.0 # Current standard deviation
        self.p_min: float = float("inf")
        self.s_min: float = float("inf")
        self.p_s_min: float = float("inf")
        self.drift_detected = False
        self.warning_detected = False

    def update(self, val: float) -> bool:
        # Binary error: 1 if error, 0 if correct
        err = 1.0 if val > 0.0 else 0.0
        self.total_samples += 1
        self.drift_detected = False
        self.warning_detected = False

        # Incremental update of error rate mean
        self.p = self.p + (err - self.p) / self.total_samples
        self.s = max(math.sqrt(max(self.p * (1.0 - self.p) / self.total_samples, 0.0)), 0.005)

        if self.total_samples < self.min_instances:
            return False

        ps = self.p + self.s

        # Update historical minimum
        if ps < self.p_s_min:
            self.p_min = self.p
            self.s_min = self.s
            self.p_s_min = ps

        # Check drift condition
        if ps > self.p_min + self.out_control_level * self.s_min:
            self.drift_detected = True
            return True
        elif ps > self.p_min + self.warning_level * self.s_min:
            self.warning_detected = True

        return False


class PageHinkleyDetector(BaseDriftDetector):
    """
    Page-Hinkley test (Page 1954).
    Monitors continuous sequential signals (such as prediction loss).
    Signals drift when the cumulative sum deviation exceeds threshold lambda.
    """
    def __init__(self, threshold: float = 50.0, delta: float = 0.005, alpha: float = 0.999):
        super().__init__()
        self.threshold = threshold
        self.delta = delta
        self.alpha = alpha
        self.reset()

    def reset(self) -> None:
        self.total_samples = 0
        self.x_mean: float = 0.0
        self.sum_val: float = 0.0
        self.min_sum: float = 0.0
        self.drift_detected = False
        self.warning_detected = False

    def update(self, val: float) -> bool:
        self.total_samples += 1
        self.drift_detected = False

        # Incremental mean
        self.x_mean = self.x_mean + (val - self.x_mean) / self.total_samples
        self.sum_val = self.alpha * self.sum_val + (val - self.x_mean - self.delta)

        if self.sum_val < self.min_sum:
            self.min_sum = self.sum_val

        ph_val = self.sum_val - self.min_sum
        if ph_val > self.threshold:
            self.drift_detected = True
            return True

        return False


class ADWINDetector(BaseDriftDetector):
    """
    Adaptive Windowing (ADWIN) detector based on Bifet & Gavaldà (2007).
    Maintains a variable-length sliding window and evaluates statistical divergence
    between sub-windows using the Hoeffding bound with confidence parameter delta.
    """
    def __init__(self, delta: float = 0.002, max_window_size: int = 200, min_subwindow_size: int = 10):
        super().__init__()
        self.delta = delta
        self.max_window_size = max_window_size
        self.min_subwindow_size = min_subwindow_size
        self.window: List[float] = []
        self.reset()

    def reset(self) -> None:
        self.window = []
        self.total_samples = 0
        self.drift_detected = False
        self.warning_detected = False

    def update(self, val: float) -> bool:
        self.total_samples += 1
        self.window.append(float(val))
        self.drift_detected = False

        if len(self.window) > self.max_window_size:
            self.window.pop(0)

        n = len(self.window)
        if n < self.min_subwindow_size * 2:
            return False

        # Test possible split points in the window
        # To remain fast on streaming data, check periodic split points
        step = max(1, n // 10)
        window_arr = np.array(self.window)

        for split in range(self.min_subwindow_size, n - self.min_subwindow_size, step):
            n0 = split
            n1 = n - split
            m = 1.0 / (1.0 / n0 + 1.0 / n1)

            mu0 = np.mean(window_arr[:split])
            mu1 = np.mean(window_arr[split:])
            diff = abs(mu0 - mu1)

            # Hoeffding bound threshold
            eps_cut = math.sqrt((1.0 / (2.0 * m)) * math.log(max(4.0 * n / self.delta, 1.1)))

            if diff > eps_cut:
                self.drift_detected = True
                # Drop older sub-window upon detected drift
                self.window = self.window[split:]
                return True

        return False


def get_detector(name: str = "adwin", **kwargs) -> BaseDriftDetector:
    """Factory function for instantiating drift detectors."""
    name_clean = name.lower().strip()
    if name_clean == "ddm":
        return DDMDetector(**kwargs)
    elif name_clean == "page_hinkley" or name_clean == "ph":
        return PageHinkleyDetector(**kwargs)
    elif name_clean == "adwin":
        return ADWINDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector: '{name}'. Supported: 'adwin', 'ddm', 'page_hinkley'")
