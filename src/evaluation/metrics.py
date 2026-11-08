from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class RoundMetrics:
    """
    Standardized per-round telemetry record for Federated Learning experiments.
    """
    round: int
    global_loss: float
    global_accuracy: float
    mean_client_train_loss: float
    mean_client_test_loss: float
    mean_client_test_accuracy: float
    std_client_test_accuracy: float
    per_client_test_accuracy: Dict[int, float]
    round_comm_bytes: int
    cumulative_comm_bytes: int
    round_duration_sec: float


class MetricsLogger:
    """
    Collects, aggregates, and exports FL experimental telemetry.
    Ensures reproducibility and exact logging to CSV and JSON formats.
    """
    def __init__(self, output_dir: str = "./results", experiment_name: str = "fedavg_baseline"):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.history: List[RoundMetrics] = []
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)

    def log_round(self, metrics: RoundMetrics) -> None:
        """Appends round metrics to history."""
        self.history.append(metrics)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts history to Pandas DataFrame (excluding per_client nested dict for flat columns)."""
        records = []
        for r in self.history:
            d = asdict(r)
            # Flatten per_client_test_accuracy
            client_accs = d.pop("per_client_test_accuracy")
            for cid, acc in client_accs.items():
                d[f"client_{cid}_test_acc"] = acc
            records.append(d)
        return pd.DataFrame(records)

    def save_results(self) -> Tuple[str, str]:
        """
        Saves full history to CSV and JSON files.
        Returns paths to saved files.
        """
        df = self.to_dataframe()
        csv_path = os.path.join(self.output_dir, "tables", f"{self.experiment_name}_metrics.csv")
        df.to_csv(csv_path, index=False)

        json_path = os.path.join(self.output_dir, "logs", f"{self.experiment_name}_full.json")
        json_data = [asdict(r) for r in self.history]
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        return csv_path, json_path

    def get_summary_table(self) -> str:
        """
        Returns a formatted table of key progress across rounds.
        Falls back to to_string if tabulate is not installed.
        """
        df = self.to_dataframe()
        summary_cols = [
            "round",
            "global_accuracy",
            "global_loss",
            "mean_client_test_accuracy",
            "round_comm_bytes",
            "round_duration_sec",
        ]
        available_cols = [c for c in summary_cols if c in df.columns]
        subset_df = df[available_cols]
        try:
            return subset_df.to_markdown(index=False)
        except (ImportError, ModuleNotFoundError):
            return subset_df.to_string(index=False)
