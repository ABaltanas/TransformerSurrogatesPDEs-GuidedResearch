# Guided Research: Neural Operators & Transformer Surrogates for Plasticity

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20in%20Progress-orange)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Benchmarking and evaluation of transformer-based surrogate models for partial differential equations (PDEs), specifically evaluating **SMART** and **uPDEt** on non-linear solid mechanics (plasticity).

---

## 🎯 Objectives & Milestones

| Milestone | Objective | Focus & Scope |
| :--- | :--- | :--- |
| **Milestone 1** | **SMART Baseline Setup** | Set up and run [SMART](https://github.com/jhagnberger/smart) on the 2D/3D plasticity dataset using local/cluster infrastructure. |
| **Milestone 2** | **uPDEt Integration** | Integrate and benchmark the **uPDEt** architecture on the same plasticity benchmark. |
| **Milestone 3** | **Comparative Analysis** | Perform comparative analysis against baselines based on the primary test case from the *PLT* paper. |

---

## 📚 References & Datasets

- **Benchmark Dataset:** Plasticity dataset from [Geo-FNO](https://github.com/neuraloperator/Geo-FNO).
- **Evaluation Baseline:** Primary test case formulated in the *PLT* paper.
- **Models:**
  - **SMART:** Scalable Mesh Autoregressive Transformer ([GitHub](https://github.com/jhagnberger/smart)).
  - **uPDEt:** Universal PDE Transformer.

| Resource | Type | Source / Repository | Purpose |
| :--- | :--- | :--- | :--- |
| **Geo-FNO** | Dataset & Reference | [neuraloperator/Geo-FNO](https://github.com/neuraloperator/Geo-FNO) | 2D/3D non-linear plasticity benchmark data |
| **SMART** | Architecture | [jhagnberger/smart](https://github.com/jhagnberger/smart) | Autoregressive transformer on unstructured meshes |
| **uPDEt** | Architecture | Reference Implementation | Universal operator transformer surrogate |
| **PLT Paper** | Baseline Standard | Literature Formulation | Benchmark comparison criteria & primary test case |

---

## 🚀 Setup & Execution

### 1. Clone with Submodules
```bash
git clone --recurse-submodules <REPO_URL>
cd <REPO_DIR>
```

### 2. Environment Setup
```bash
pip install -r requirements.txt
```

### 3. Data Placement
Download the Geo-FNO plasticity dataset and place the raw files into `data/raw/` (ignored by git):
```bash
mkdir -p data/raw
# Place downloaded plasticity dataset files into data/raw/
```

> [!NOTE]
> The `data/` and `results/` folders are untracked by Git via `.gitignore` to prevent committing heavy datasets and runtime artifacts.
---

# Repository Structure

```
├── configs/                 # Configuration files for hyperparameters and experiment setups
├── data/                    # [gitignore] Storage for raw and preprocessed datasets
│   ├── raw/                 # Original, immutable raw datasets and benchmark inputs
│   └── processed/           # Cleaned, transformed, and model-ready datasets
├── external/                # Third-party repositories and submodules (e.g., SMART)
├── scripts/                 # Standalone execution scripts for training, evaluation, and plotting
├── src/                     # Reusable library modules (dataset, metrics, model adapters)
├── results/                 # [gitignore] Experiment outputs, checkpoints, and generated figures
├── .agents/                 # AI agent skills and project configurations
├── .gitignore               # Files and patterns ignored by Git version control
├── .gitmodules              # Submodule configuration pointers
├── AGENT.md                 # Project guidelines and development context
├── README.md                # Project documentation and repository structure overview
└── requirements.txt         # Project package dependencies and environment specifications
```
