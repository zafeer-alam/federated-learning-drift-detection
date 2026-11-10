import argparse
import logging
import os
import sys
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.clients.client import FLClient
from src.communication.scheduler import get_scheduler
from src.drift.detector import get_detector
from src.drift.drift_simulator import (
    DriftSimulator,
    create_abrupt_drift_schedule,
    create_gradual_drift_schedule,
    create_recurring_drift_schedule,
    create_staggered_drift_schedule,
)
from src.evaluation.metrics import MetricsLogger, RoundMetrics
from src.models.model_registry import ModelPool, ModelSnapshot
from src.selection.model_selector import get_model_selector
from src.server.server import FLServer
from src.utils.data import load_federated_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_single_sweep(
    exp_id: str,
    drift_type: str = "abrupt",
    selector_type: str = "drift_aware",
    scheduler_type: str = "fixed",
    detector_type: str = "ddm",
    label_delay: int = 0,
    enable_pool: bool = True,
    num_rounds: int = 4,
    drift_round: int = 2,
    num_clients: int = 3,
    alpha: float = 0.5,
    seed: int = 42,
    output_dir: str = "./results",
) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    experiment_name = f"sweep_{exp_id}_{drift_type}_{detector_type}_sel_{selector_type}_sched_{scheduler_type}_pool_{enable_pool}_del_{label_delay}_s{seed}"
    logger.info(f"--> Launching Sweep: {experiment_name}")

    client_splits, global_test_loader = load_federated_dataset(
        dataset_name="mnist",
        num_clients=num_clients,
        partition_strategy="dirichlet",
        dirichlet_alpha=alpha,
        batch_size=64,
        seed=seed,
    )

    server = FLServer(architecture="cnn", num_classes=10)
    clients: List[FLClient] = [
        FLClient(
            client_id=split.client_id,
            train_loader=split.train_loader,
            test_loader=split.test_loader,
            architecture="cnn",
        )
        for split in client_splits
    ]

    drift_sim = DriftSimulator(num_classes=10, seed=seed)
    client_ids = list(range(num_clients))
    if drift_type == "abrupt":
        schedules = create_abrupt_drift_schedule(client_ids=client_ids, drift_round=drift_round)
    elif drift_type == "gradual":
        schedules = create_gradual_drift_schedule(
            client_ids=client_ids, start_round=drift_round, end_round=min(drift_round + 2, num_rounds)
        )
    elif drift_type == "recurring":
        schedules = create_recurring_drift_schedule(
            client_ids=client_ids, recurrence_intervals=[(drift_round, drift_round + 1)]
        )
    elif drift_type == "staggered":
        onset_map = {i: drift_round + i for i in range(num_clients)}
        schedules = create_staggered_drift_schedule(client_onset_map=onset_map)
    else:
        schedules = []

    for ev in schedules:
        drift_sim.register_schedule(ev)

    schedulers = {i: get_scheduler(scheduler_type) for i in range(num_clients)}
    detectors = {i: get_detector(detector_type) for i in range(num_clients)}
    model_pools = {i: ModelPool(max_size=5) for i in range(num_clients)}
    model_selectors = {i: get_model_selector(selector_type) for i in range(num_clients)}

    metrics_logger = MetricsLogger(output_dir=output_dir, experiment_name=experiment_name)
    cumulative_comm_bytes = 0
    detector_history = []
    round_accs = []

    for r in range(1, num_rounds + 1):
        round_comm_bytes = 0
        client_updates = []
        client_train_losses = []
        detector_alarms: Dict[int, bool] = {}

        global_params = server.get_parameters()

        for client in clients:
            cid = client.client_id

            drifted_train_loader = drift_sim.wrap_loader(cid, r, client.train_loader)
            drifted_test_loader = drift_sim.wrap_loader(cid, r, client.test_loader)
            client.train_loader = drifted_train_loader
            client.test_loader = drifted_test_loader

            pre_eval = client.evaluate(global_params)
            error_rate = 1.0 - pre_eval["accuracy"]

            # Simulate label delay: if delay > 0 and r < drift_round + delay, suppress immediate alarm
            effective_alarm = False
            if label_delay > 0 and r < (drift_round + label_delay):
                effective_alarm = False
            else:
                if detector_type.lower() == "ddm":
                    effective_alarm = detectors[cid].update(error_rate)
                else:
                    effective_alarm = detectors[cid].update(pre_eval["loss"])

            detector_alarms[cid] = effective_alarm

            should_sync = schedulers[cid].should_communicate(cid, r, detector_alarm=effective_alarm)

            if should_sync:
                client.set_parameters(global_params)
                round_comm_bytes += server.model_size_bytes

            updated_params, num_samples, train_metrics = client.train(epochs=1, lr=0.01)
            client_train_losses.append(train_metrics["train_loss"])

            if should_sync:
                client_updates.append((updated_params, num_samples))
                round_comm_bytes += server.model_size_bytes

            if enable_pool and pre_eval["accuracy"] > 0.80:
                model_pools[cid].add(
                    ModelSnapshot(
                        name=f"c{cid}_r{r}",
                        parameters=client.get_parameters(),
                        round_created=r,
                        accuracy=pre_eval["accuracy"],
                    )
                )

            def eval_candidate(params):
                return client.evaluate(params)["accuracy"]

            active_pool = model_pools[cid] if enable_pool else None
            selected_params, reason = model_selectors[cid].select(
                client_id=cid,
                round_idx=r,
                global_params=global_params,
                local_params=client.get_parameters(),
                pool=active_pool,
                detector_alarm=effective_alarm,
                eval_fn=eval_candidate,
            )
            client.set_parameters(selected_params)

        if client_updates:
            server.aggregate(client_updates)

        cumulative_comm_bytes += round_comm_bytes
        global_eval = server.evaluate(global_test_loader)
        client_evals = [client.evaluate() for client in clients]
        mean_client_acc = float(np.mean([e["accuracy"] for e in client_evals]))
        mean_client_loss = float(np.mean([e["loss"] for e in client_evals]))

        round_accs.append(mean_client_acc)
        detector_history.append(sum(detector_alarms.values()))

        rm = RoundMetrics(
            round=r,
            global_loss=global_eval["loss"],
            global_accuracy=global_eval["accuracy"],
            mean_client_train_loss=float(np.mean(client_train_losses)),
            mean_client_test_loss=mean_client_loss,
            mean_client_test_accuracy=mean_client_acc,
            std_client_test_accuracy=float(np.std([e["accuracy"] for e in client_evals])),
            per_client_test_accuracy={c.client_id: e["accuracy"] for c, e in zip(clients, client_evals)},
            round_comm_bytes=round_comm_bytes,
            cumulative_comm_bytes=cumulative_comm_bytes,
            round_duration_sec=0.0,
        )
        metrics_logger.log_round(rm)

    metrics_logger.save_results()

    return {
        "exp_id": exp_id,
        "drift_type": drift_type,
        "detector": detector_type,
        "selector": selector_type,
        "scheduler": scheduler_type,
        "pool": enable_pool,
        "label_delay": label_delay,
        "final_client_acc": round_accs[-1],
        "min_client_acc": min(round_accs),
        "post_drift_acc": round_accs[drift_round] if drift_round < len(round_accs) else round_accs[-1],
        "total_alarms": sum(detector_history),
        "total_comm_mb": cumulative_comm_bytes / (1024 * 1024),
    }


def run_all_extended_sweeps():
    logger.info("=== STARTING EXTENDED EXPERIMENTAL SWEEPS (EXPS 2, 3, 4) ===")
    records = []

    # Experiment 2: Full Drift Taxonomy Sweep across Detectors (Gradual, Recurring, Staggered)
    for dtype in ["gradual", "recurring", "staggered"]:
        for det in ["ddm", "page_hinkley", "adwin"]:
            res = run_single_sweep(
                exp_id=f"exp2_{dtype}_{det}",
                drift_type=dtype,
                detector_type=det,
                selector_type="drift_aware",
                scheduler_type="fixed",
                num_rounds=3,
                drift_round=2,
            )
            records.append(res)

    # Experiment 3: Label Availability (Immediate vs Delayed Feedback)
    for delay in [0, 1, 2]:
        res = run_single_sweep(
            exp_id=f"exp3_label_delay_{delay}",
            drift_type="abrupt",
            detector_type="ddm",
            selector_type="drift_aware",
            scheduler_type="fixed",
            label_delay=delay,
            num_rounds=3,
            drift_round=2,
        )
        records.append(res)

    # Experiment 4: Model Retention Pool vs No Pool under Recurring Drift
    for pool_flag in [True, False]:
        res = run_single_sweep(
            exp_id=f"exp4_pool_{pool_flag}",
            drift_type="recurring",
            detector_type="ddm",
            selector_type="drift_aware",
            scheduler_type="fixed",
            enable_pool=pool_flag,
            num_rounds=3,
            drift_round=2,
        )
        records.append(res)

    df_summary = pd.DataFrame(records)
    summary_path = "./results/tables/extended_sweeps_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"=== ALL EXTENDED SWEEPS COMPLETE. Summary saved to {summary_path} ===")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    run_all_extended_sweeps()
