# Evaluating the Value of Drift Detection for Model Selection in Federated Learning

> **Project type:** Research / Experimental Machine Learning  
> **Primary focus:** Concept drift, personalized federated learning, per-client model selection, and communication scheduling

## 1. Project Overview

Federated Learning (FL) enables multiple clients to collaboratively train machine learning models without sharing their private local data. In real-world environments, however, client data distributions can change over time. These changes, commonly called **concept drift**, may reduce model performance and limit the effectiveness of conventional federated learning approaches.

Personalized Federated Learning (PFL) addresses differences between clients by maintaining client-specific models. A remaining challenge is deciding when a client should keep its current model, switch to a previously trained model, or adapt its model to new data.

This project investigates whether **explicit drift detection** improves per-client model selection and communication scheduling compared with fixed strategies, particularly when clients experience changing data distributions.

## 2. Research Question

**Does explicit drift detection improve per-client model selection and communication scheduling when client data distributions change over time?**

## 3. Research Objectives

1. Implement baseline Federated Learning approaches using FedAvg and personalized models under non-IID client data.
2. Simulate abrupt, gradual, recurring, and staggered changes in client data distributions.
3. Compare fixed model selection with drift-detection-based model selection under matched information and resource budgets.
4. Investigate the effects of label availability, model retention, and communication scheduling.
5. Evaluate predictive accuracy, adaptation delay, forgetting, false alarms, and communication cost.
6. Determine the conditions under which the benefits of drift detection justify its additional computational and communication costs.

## 4. Proposed System

The experimental system will contain multiple clients coordinated by a central federated server.

### Main components

- **Clients:** Hold private local datasets, train local models, and monitor their local data or prediction streams.
- **Federated server:** Coordinates training and aggregates client updates.
- **Drift detector:** Identifies evidence of a change in a client's data distribution or monitored model performance.
- **Model selection module:** Decides whether to retain the current model, switch to a retained model, or adapt a model.
- **Communication scheduler:** Determines when a client should send updates or request model information from the server.

### High-level workflow

1. Initialize clients with non-IID local data.
2. Train baseline federated and personalized models.
3. Introduce controlled distribution shifts over time.
4. Monitor client streams and run the selected drift detector.
5. On a drift signal, evaluate available models and select a model or trigger adaptation.
6. Apply the communication policy being tested.
7. Record performance, detection events, model switches, and resource usage.

## 5. Project Methodology

### Phase 1 — Literature Review and Problem Formulation

- Study Federated Averaging (FedAvg) and personalized federated learning.
- Review IID and non-IID data partitioning methods.
- Study concept drift and online drift detection.
- Review model selection, model retention, and communication-efficient FL.
- Define the experimental assumptions, baselines, and evaluation protocol.

**Deliverable:** Literature review and finalized experimental design.

### Phase 2 — Baseline Federated Learning Implementation

- Set up the Python and deep learning environment.
- Select and prepare an initial classification dataset.
- Partition the data across simulated clients using a non-IID strategy.
- Implement FedAvg.
- Implement a personalized baseline in which clients maintain client-specific models.
- Record baseline performance over communication rounds.

**Initial baseline candidates:**

| Method | Description |
|---|---|
| FedAvg | Clients train locally and the server aggregates model updates into a global model. |
| Local training | Each client trains independently without server aggregation. |
| Personalized FL | Clients retain client-specific models or personalized parameters. |

**Deliverable:** Reproducible baseline implementations and logs.

### Phase 3 — Concept Drift Simulation

Simulate four types of changes in client data distributions:

1. **Abrupt drift:** A client's data distribution changes suddenly at a specified time.
2. **Gradual drift:** The old distribution is progressively replaced by a new distribution.
3. **Recurring drift:** A previously observed distribution returns after an interval.
4. **Staggered drift:** Different clients experience drift at different times.

The simulation should record the true drift time and type for each client. This ground truth will be used to evaluate detection performance and adaptation delay.

**Deliverable:** Configurable drift scenarios and reproducible data streams.

### Phase 4 — Drift Detection and Model Selection

Start with one detector and add alternatives only after the end-to-end pipeline works.

Candidate detectors:

- **ADWIN:** Maintains an adaptive window and detects changes in a monitored stream.
- **DDM:** Uses changes in prediction error to identify possible deterioration.
- **Page-Hinkley:** Detects changes in a monitored sequence, such as prediction loss.

The drift-aware model-selection workflow will be:

1. Monitor a suitable statistic at each client.
2. Run the drift detector on the monitored stream.
3. If no drift is signaled, retain the current model.
4. If drift is signaled, evaluate the current model and available alternatives using the information allowed by the experiment.
5. Select a retained model or trigger local adaptation.
6. Record the decision, detection time, and resulting performance.

Compare at least these policies:

- **Fixed selection:** Selection follows a predefined policy and is not triggered by an explicit drift alarm.
- **Drift-aware selection:** A drift alarm triggers model evaluation and may cause switching or adaptation.

**Deliverable:** Drift detector integration and per-client model-selection policies.

### Phase 5 — Model Retention and Communication Scheduling

#### Model retention

Evaluate whether clients benefit from keeping previously trained models that may become useful again when a distribution recurs.

Suggested configurations:

- No model retention: keep only the currently selected model.
- Limited retention: keep a small, fixed number of previous models.
- Multiple-model retention: keep a larger candidate pool subject to a defined memory budget.

For recurring drift, evaluate whether selecting a retained model is more effective than adapting a model from scratch.

#### Communication scheduling

Compare:

- Fixed communication intervals.
- Drift-triggered communication.
- Periodic communication combined with drift-triggered updates.

Measure communication in clearly defined units, such as bytes transferred, number of messages, or model updates. Keep the accounting consistent across methods.

**Deliverable:** Model-retention and communication-scheduling experiments.

### Phase 6 — Evaluation and Analysis

Run controlled experiments across drift types and policy configurations. Use matched information and resource budgets wherever possible. In particular, document label access, local update opportunities, model memory, computation, and communication budgets for every method.

**Deliverable:** Results, plots, ablation studies, and interpretation.

## 6. Experimental Design

| Experiment | Comparison or factor | Main question |
|---|---|---|
| 1 | Fixed selection vs. drift-aware selection | Does explicit detection improve model selection? |
| 2 | Abrupt, gradual, recurring, and staggered drift | Which drift types benefit most? |
| 3 | Different label-availability settings | How does access to labels affect detection and adaptation? |
| 4 | With vs. without model retention | Does retaining models help, especially under recurring drift? |
| 5 | Fixed vs. drift-triggered communication | Can communication be reduced without unacceptable performance loss? |
| 6 | Component ablations | Which parts of the proposed system contribute to the observed results? |

### Fair-comparison requirements

- Use the same client data streams and drift schedules for competing methods.
- Match or explicitly report label access and information available to each method.
- Match or report local training opportunities and computation budgets.
- Define model-memory limits for retention methods.
- Count all communication caused by drift detection, model evaluation, switching, and training.
- Repeat experiments with multiple random seeds and report variability.

## 7. Datasets

Begin with a small, manageable dataset and extend only after the pipeline is stable.

| Dataset | Task | Intended use |
|---|---|---|
| MNIST | Handwritten digit classification | Initial implementation and debugging |
| Fashion-MNIST | Clothing image classification | Secondary validation on a more challenging dataset |
| CIFAR-10 | Natural image classification | Optional extension if time and compute permit |

**Recommended starting point:** MNIST with approximately 5–10 simulated clients. The exact client count, model architecture, partitioning method, and drift schedule should be configurable rather than hard-coded.

## 8. Evaluation Metrics

| Metric | Definition / purpose |
|---|---|
| Predictive accuracy | Measures prediction quality over time and after drift. |
| Adaptation delay | Time or number of examples/rounds between true drift and recovery to a defined performance threshold. |
| Forgetting | Measures degradation on previously encountered distributions or tasks after later adaptation. |
| False alarm rate | Measures drift alarms that do not correspond to a simulated drift event, using a documented matching window. |
| Communication cost | Measures bytes, messages, or model updates exchanged. |
| Detection delay | Time or number of examples/rounds between true drift and the detector's alarm. |
| Model-switch frequency | Counts model switches, helping identify unstable or excessive switching. |
| Computational overhead | Measures additional runtime or processing caused by detection and model evaluation. |

Report both aggregate metrics and per-client results where possible. Accuracy should also be plotted over time so that temporary performance drops and recovery behavior are visible.

## 9. Technology Stack

| Component | Suggested technology |
|---|---|
| Language | Python |
| Deep learning | PyTorch |
| Federated learning simulation | Flower or a lightweight custom simulator |
| Drift detection | River |
| Data processing | NumPy, Pandas, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Experiment tracking | CSV logs initially; TensorBoard or another tracker if useful |
| Development | Jupyter Notebook and/or VS Code |

These are proposed tools, not strict requirements. Keep the implementation modular so that the FL framework or detector can be changed without rewriting the experiment logic.

## 10. Suggested Repository Structure

```text
federated-learning-drift-detection/
├── README.md
├── requirements.txt
├── configs/
│   ├── baseline.yaml
│   ├── drift_scenarios.yaml
│   └── experiments.yaml
├── data/
│   └── README.md
├── src/
│   ├── clients/
│   │   ├── client.py
│   │   └── client_state.py
│   ├── server/
│   │   └── server.py
│   ├── models/
│   │   ├── model.py
│   │   └── model_registry.py
│   ├── federated/
│   │   ├── fedavg.py
│   │   └── personalized.py
│   ├── drift/
│   │   ├── detector.py
│   │   └── drift_simulator.py
│   ├── selection/
│   │   └── model_selector.py
│   ├── communication/
│   │   └── scheduler.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate.py
│   └── utils/
│       ├── data.py
│       └── logging.py
├── experiments/
│   ├── run_baseline.py
│   ├── run_drift.py
│   └── run_ablation.py
├── results/
│   ├── logs/
│   ├── figures/
│   └── tables/
└── notebooks/
    └── exploratory_analysis.ipynb
```

This is a suggested structure. It is acceptable to begin with fewer files and split modules as the implementation grows.

## 11. Four-Week Execution Plan

| Week | Tasks | Expected deliverable |
|---|---|---|
| 1 — Baselines | Literature review, environment setup, dataset partitioning, FedAvg, personalized baseline | Working FL baseline |
| 2 — Drift | Drift simulation, initial detector integration, detection logging | Dynamic FL environment with drift detection |
| 3 — Proposed mechanisms | Fixed and drift-aware selection, model retention, communication scheduling | End-to-end comparison pipeline |
| 4 — Evaluation | Run experiments, ablations, plots, analysis, report and presentation | Research prototype and final report |

The four-week schedule is a target for a focused prototype. A complete and statistically robust study may require additional time, especially for repeated runs and broader datasets.

## 12. Initial Scope and Priorities

To avoid making the first implementation too large, use this order:

1. MNIST and a small number of simulated clients.
2. FedAvg and one personalized baseline.
3. Non-IID client partitioning.
4. Abrupt and gradual drift first.
5. One drift detector, initially ADWIN or DDM depending on the monitored signal and label availability.
6. Fixed versus drift-aware model selection.
7. Add model retention and communication scheduling.
8. Extend to recurring and staggered drift, then evaluate label availability and additional detectors.

Do not claim that drift detection improves performance before experiments establish it. A finding that detection provides little benefit under some conditions is also a valid research result.

## 13. Initial Mathematical Formulation

Let:

- \(K\) be the number of clients.
- \(D_k^t\) be the data distribution of client \(k\) at time \(t\).
- \(M_k^t\) be the model selected by client \(k\) at time \(t\).
- \(d_k^t\) be the drift-detector signal.
- \(C^t\) be the total communication cost at time \(t\).
- \(F^t\) be a forgetting measure at time \(t\).
- \(A^t\) be predictive performance at time \(t\).

A simplified model-selection rule is:

\[
M_k^t =
\begin{cases}
M_k^{t-1}, & d_k^t = 0,\\
\displaystyle\arg\max_{M\in\mathcal{M}_k} \widehat{A}_k(M,t), & d_k^t = 1,
\end{cases}
\]

where \(\mathcal{M}_k\) is the set of models available to client \(k\), and \(\widehat{A}_k(M,t)\) is an estimate of model performance using only information permitted by the experimental setting.

A possible high-level objective is:

\[
\min_{\pi}\; \mathbb{E}\left[\sum_{t=1}^{T}
\left(1-A^t + \lambda C^t + \mu F^t\right)\right],
\]

where \(\pi\) represents the model-selection and communication policy, and \(\lambda\) and \(\mu\) control the trade-offs. This is an initial conceptual formulation; the final objective and metric definitions should be fixed after the experimental protocol is specified.

## 14. Expected Outcomes

The project aims to determine:

- Whether explicit drift detection improves per-client model selection over fixed strategies.
- Whether retained models improve recovery under recurring drift.
- Whether drift-triggered communication reduces communication cost while preserving useful predictive performance.
- How label availability changes the value and practicality of drift detection.
- Which drift patterns benefit most from explicit detection.
- When the additional detection, computation, memory, and communication costs are justified.

## 15. Final Deliverables

- [ ] Literature review and finalized problem formulation.
- [ ] Reproducible FedAvg baseline.
- [ ] Personalized FL baseline.
- [ ] Non-IID client data simulation.
- [ ] Abrupt, gradual, recurring, and staggered drift scenarios.
- [ ] Drift detection module.
- [ ] Fixed and drift-aware model-selection policies.
- [ ] Model-retention mechanism.
- [ ] Communication-scheduling policies.
- [ ] Evaluation metrics and experiment logs.
- [ ] Comparative experiments and ablation studies.
- [ ] Figures, tables, and result analysis.
- [ ] Final research report and presentation.

## 16. Guidance for AI-Assisted Development in Antigravity

Use this README as the project specification. Implement the project incrementally rather than generating the entire system in one step.

Recommended implementation order:

1. Inspect the repository and propose a minimal implementation plan before editing files.
2. Set up dependencies and a reproducible configuration system.
3. Implement and test data loading and non-IID client partitioning.
4. Implement the baseline FL simulator and verify its training loop.
5. Add drift simulation with deterministic, configurable drift schedules.
6. Add a detector behind a common interface.
7. Implement fixed and drift-aware model selection as separate policies.
8. Add model retention and communication scheduling as independently configurable components.
9. Implement metric logging and reproducible experiment scripts.
10. Run tests and small experiments before scaling up.

### Development constraints

- Keep components modular and configurable.
- Use fixed random seeds for reproducibility, while supporting multiple seeds for final evaluation.
- Do not silently change the research question or experimental assumptions.
- Do not fabricate results or claim experiments passed unless they were actually run.
- Log true drift times separately from detector alarms.
- Make label access explicit in each experiment configuration.
- Count communication and computation overhead consistently across methods.
- Add unit tests for drift schedules, detector behavior, model selection, and metric calculations.
- Prefer a small working end-to-end experiment over a large unfinished implementation.

## 17. References and Further Reading

The project should build a literature review around foundational work in:

- Federated Averaging and communication-efficient federated optimization.
- Personalized federated learning and client heterogeneity.
- Concept drift in data streams.
- Online drift detectors such as ADWIN, DDM, and Page-Hinkley.
- Continual learning, model reuse, and catastrophic forgetting.

Add complete bibliographic references after selecting and verifying the specific papers used in the literature review. Do not cite papers that have not been checked.

---

**Project status:** Planning  
**Immediate next milestone:** Build and validate the non-IID FedAvg baseline on MNIST before implementing drift detection.
