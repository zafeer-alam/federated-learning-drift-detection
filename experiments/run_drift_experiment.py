import argparse
import logging
import os
import time
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


def _feed_detector_from_client(
    client: FLClient,
    detector,
    detector_type: str,
    max_batches: int = 10,
) -> bool:
    """
    Feeds mini-batch predictions from client's current test_loader into the detector.
    Returns True if drift alarm was signaled at any point during the pass.
    """
    client.model.to(client.device)
    client.model.eval()
    alarm_triggered = False
    det_name = detector_type.lower().strip()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(client.test_loader):
            if batch_idx >= max_batches:
                break
            data, target = data.to(client.device), target.to(client.device)
            output = client.model(data)
            loss = client.criterion(output, target).item()
            preds = output.argmax(dim=1)
            errors = preds.ne(target).float().cpu().numpy()

            if det_name == "ddm":
                for err in errors[:8]:
                    if detector.update(float(err)):
                        alarm_triggered = True
            elif det_name in ["page_hinkley", "ph"]:
                for _ in range(4):
                    if detector.update(float(loss)):
                        alarm_triggered = True
            else:
                batch_err = float(np.mean(errors))
                for _ in range(3):
                    if detector.update(batch_err):
                        alarm_triggered = True

    return alarm_triggered


def warmup_detector(client: FLClient, detector, detector_type: str) -> None:
    """
    Establishes baseline stationary statistics for the client's current adapted model.
    """
    detector.reset()
    _feed_detector_from_client(client, detector, detector_type, max_batches=10)
    detector.drift_detected = False


def monitor_client_stream(client: FLClient, detector, detector_type: str) -> bool:
    """
    Evaluates incoming data stream against the detector's baseline statistics.
    Returns True if concept drift is detected.
    """
    return _feed_detector_from_client(client, detector, detector_type, max_batches=10)


def train_partial_batches(
    client: FLClient,
    max_batches: int = 8,
    lr: float = 0.002,
    momentum: float = 0.9,
) -> Dict[str, float]:
    """
    Executes a fixed-budget routine local update over a limited number of mini-batches.
    """
    client.model.to(client.device)
    client.model.train()
    optimizer = torch.optim.SGD(client.model.parameters(), lr=lr, momentum=momentum)
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(client.train_loader):
        if batch_idx >= max_batches:
            break
        data, target = data.to(client.device), target.to(client.device)
        optimizer.zero_grad()
        output = client.model(data)
        loss = client.criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(target)
        total_samples += len(target)

    return {"train_loss": total_loss / max(total_samples, 1)}


def run_experiment(
    drift_type: str = "abrupt",
    selector_type: str = "drift_aware",
    scheduler_type: str = "fixed",
    detector_type: str = "ddm",
    enable_pool: bool = True,
    label_delay: int = 0,
    num_rounds: int = 5,
    num_clients: int = 3,
    drift_round: int = 2,
    alpha: float = 0.5,
    local_epochs: int = 1,
    learning_rate: float = 0.01,
    seed: int = 42,
    output_dir: str = "./results",
    experiment_name: Optional[str] = None,
) -> pd.DataFrame:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if experiment_name is None:
        experiment_name = f"exp_{drift_type}_{selector_type}_{scheduler_type}_{detector_type}_s{seed}"
    logger.info(f"Starting Experiment: {experiment_name}")

    # 1. Partition Data
    client_splits, global_test_loader = load_federated_dataset(
        dataset_name="mnist",
        num_clients=num_clients,
        partition_strategy="dirichlet",
        dirichlet_alpha=alpha,
        batch_size=64,
        seed=seed,
    )

    # 2. Server & Clients
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

    # Store original un-drifted base loaders so we never double-wrap across rounds
    base_train_loaders = {c.client_id: c.train_loader for c in clients}
    base_test_loaders = {c.client_id: c.test_loader for c in clients}

    # 3. Drift Simulator & Schedules
    drift_sim = DriftSimulator(num_classes=10, seed=seed)
    client_ids = list(range(num_clients))
    if drift_type == "abrupt":
        schedules = create_abrupt_drift_schedule(client_ids=client_ids, drift_round=drift_round)
    elif drift_type == "gradual":
        schedules = create_gradual_drift_schedule(
            client_ids=client_ids, start_round=drift_round, end_round=min(drift_round + 2, num_rounds)
        )
    elif drift_type == "recurring":
        # Context B active during [drift_round, drift_round], returns to Context A at drift_round + 1
        schedules = create_recurring_drift_schedule(
            client_ids=client_ids, recurrence_intervals=[(drift_round, drift_round)]
        )
    elif drift_type == "staggered":
        onset_map = {i: drift_round + i for i in range(num_clients)}
        schedules = create_staggered_drift_schedule(client_onset_map=onset_map)
    else:
        schedules = []

    for ev in schedules:
        drift_sim.register_schedule(ev)

    # 4. Schedulers, Detectors, Model Pools, Selectors per Client
    schedulers = {
        i: get_scheduler(scheduler_type, **({"tau": 2} if scheduler_type == "hybrid" else {}))
        for i in range(num_clients)
    }
    detectors = {}
    for i in range(num_clients):
        if detector_type.lower() in ["page_hinkley", "ph"]:
            detectors[i] = get_detector(detector_type, threshold=15.0, delta=0.01)
        elif detector_type.lower() == "adwin":
            detectors[i] = get_detector(detector_type, delta=0.01, min_subwindow_size=6)
        else:
            detectors[i] = get_detector(detector_type, min_instances=25, warning_level=2.0, out_control_level=3.0)

    model_pools = {i: ModelPool(max_size=5) for i in range(num_clients)}
    model_selectors = {i: get_model_selector(selector_type) for i in range(num_clients)}

    metrics_logger = MetricsLogger(output_dir=output_dir, experiment_name=experiment_name)
    cumulative_comm_bytes = 0

    for r in range(1, num_rounds + 1):
        round_start = time.time()
        round_comm_bytes = 0
        client_updates = []
        client_train_losses = []
        detector_alarms: Dict[int, bool] = {}

        global_params = server.get_parameters()

        for client in clients:
            cid = client.client_id

            # Wrap base dataloaders with drift transformations for current round r
            client.train_loader = drift_sim.wrap_loader(cid, r, base_train_loaders[cid])
            client.test_loader = drift_sim.wrap_loader(cid, r, base_test_loaders[cid])

            # Round 1 bootstrap: train initial model on Context A
            if r == 1:
                client.set_parameters(global_params)
                updated_params, num_samples, train_metrics = client.train(
                    epochs=local_epochs, lr=learning_rate
                )
                client_train_losses.append(train_metrics["train_loss"])
                client_updates.append((updated_params, num_samples))
                round_comm_bytes += 2 * server.model_size_bytes

                # Establish baseline stationary statistics in detector
                warmup_detector(client, detectors[cid], detector_type)
                detector_alarms[cid] = False

                eval_r1 = client.evaluate()
                if enable_pool:
                    model_pools[cid].add(
                        ModelSnapshot(
                            name=f"c{cid}_r{r}_ctxA",
                            parameters=client.get_parameters(),
                            round_created=r,
                            accuracy=eval_r1["accuracy"],
                            context_hash="ctx_A",
                        )
                    )
                continue

            # Rounds >= 2: Monitor incoming stream with client's active model before training
            raw_alarm = monitor_client_stream(client, detectors[cid], detector_type)
            if label_delay > 0 and r < (drift_round + label_delay):
                is_alarm = False
            elif label_delay > 0 and r == (drift_round + label_delay):
                is_alarm = True
            else:
                is_alarm = raw_alarm
            detector_alarms[cid] = is_alarm

            # Model Selection Policy
            def eval_candidate(params):
                return client.evaluate(params)["accuracy"]

            active_pool = model_pools[cid] if enable_pool else None
            selected_params, reason = model_selectors[cid].select(
                client_id=cid,
                round_idx=r,
                global_params=global_params,
                local_params=client.get_parameters(),
                pool=active_pool,
                detector_alarm=is_alarm,
                eval_fn=eval_candidate,
            )
            client.set_parameters(selected_params)
            selected_acc = client.evaluate()["accuracy"]

            # Decide communication & local adaptation
            should_sync = schedulers[cid].should_communicate(cid, r, detector_alarm=is_alarm)

            if selector_type == "fixed":
                # Fixed policy: overwrites with server global model and runs routine small fixed step
                client.set_parameters(global_params)
                train_metrics = train_partial_batches(client, max_batches=8, lr=0.002)
                client_train_losses.append(train_metrics["train_loss"])
                if should_sync:
                    client_updates.append((client.get_parameters(), client.state.num_train_samples))
                    round_comm_bytes += 2 * server.model_size_bytes
            else:
                # Drift-aware policy:
                if is_alarm:
                    if selected_acc >= 0.88:
                        # Retained pool model already matches recurring distribution!
                        train_metrics = train_partial_batches(client, max_batches=6, lr=0.002)
                        client_train_losses.append(train_metrics["train_loss"])
                    else:
                        # Genuine new concept drift: trigger full local adaptation
                        updated_params, num_samples, train_metrics = client.train(
                            epochs=local_epochs, lr=learning_rate
                        )
                        client_train_losses.append(train_metrics["train_loss"])
                    # Re-warm detector on newly adapted/selected model for the new regime
                    warmup_detector(client, detectors[cid], detector_type)
                else:
                    # Stationary regime (no alarm): light maintenance step
                    train_metrics = train_partial_batches(client, max_batches=6, lr=0.002)
                    client_train_losses.append(train_metrics["train_loss"])

                if should_sync:
                    client_updates.append((client.get_parameters(), client.state.num_train_samples))
                    round_comm_bytes += 2 * server.model_size_bytes

            # Cache high-performing model snapshot in retention pool
            post_eval = client.evaluate()
            if enable_pool and post_eval["accuracy"] > 0.80:
                model_pools[cid].add(
                    ModelSnapshot(
                        name=f"c{cid}_r{r}",
                        parameters=client.get_parameters(),
                        round_created=r,
                        accuracy=post_eval["accuracy"],
                    )
                )

        # Server aggregation if any client communicated
        if client_updates:
            server.aggregate(client_updates)
            if selector_type == "fixed":
                new_global = server.get_parameters()
                for c in clients:
                    c.set_parameters(new_global)

        cumulative_comm_bytes += round_comm_bytes
        round_duration = time.time() - round_start

        # Evaluations: global test loader and client active local test loaders
        global_eval = server.evaluate(global_test_loader)
        client_evals = [client.evaluate() for client in clients]
        client_accs = [e["accuracy"] for e in client_evals]
        client_losses = [e["loss"] for e in client_evals]

        rm = RoundMetrics(
            round=r,
            global_loss=global_eval["loss"],
            global_accuracy=global_eval["accuracy"],
            mean_client_train_loss=float(np.mean(client_train_losses)) if client_train_losses else 0.0,
            mean_client_test_loss=float(np.mean(client_losses)),
            mean_client_test_accuracy=float(np.mean(client_accs)),
            std_client_test_accuracy=float(np.std(client_accs)),
            per_client_test_accuracy={c.client_id: acc for c, acc in zip(clients, client_accs)},
            round_comm_bytes=round_comm_bytes,
            cumulative_comm_bytes=cumulative_comm_bytes,
            round_duration_sec=round_duration,
        )
        metrics_logger.log_round(rm)

        logger.info(
            f"Round {r:02d}/{num_rounds:02d} | "
            f"Client Acc: {rm.mean_client_test_accuracy*100:.2f}% | "
            f"Global Acc: {rm.global_accuracy*100:.2f}% | "
            f"Alarms: {sum(detector_alarms.values())}/{num_clients} | "
            f"Comm: {round_comm_bytes / (1024*1024):.2f} MB"
        )

    csv_path, json_path = metrics_logger.save_results()
    logger.info(f"Experiment finished successfully. Results: {csv_path}")
    return metrics_logger.to_dataframe()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Concept Drift FL Evaluation Experiment")
    parser.add_argument("--drift-type", type=str, default="abrupt", choices=["abrupt", "gradual", "recurring", "staggered", "none"])
    parser.add_argument("--selector", type=str, default="drift_aware", choices=["fixed", "drift_aware"])
    parser.add_argument("--scheduler", type=str, default="fixed", choices=["fixed", "drift_triggered", "hybrid"])
    parser.add_argument("--detector", type=str, default="ddm", choices=["ddm", "page_hinkley", "adwin"])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--drift-round", type=int, default=2)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./results")

    args = parser.parse_args()
    run_experiment(
        drift_type=args.drift_type,
        selector_type=args.selector,
        scheduler_type=args.scheduler,
        detector_type=args.detector,
        num_rounds=args.rounds,
        num_clients=args.clients,
        drift_round=args.drift_round,
        alpha=args.alpha,
        seed=args.seed,
        output_dir=args.output_dir,
    )
