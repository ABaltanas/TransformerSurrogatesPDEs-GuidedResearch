# Project: Guided Research - Transformer Surrogates for PDEs

## Project Context
This repository contains the codebase for a Guided Research project at the Technical University of Munich (Chair of Physics-based Simulation, Prof. Dr.-Ing. Nils Thürey, in coordination with supervisor Luis).
- **Goal:** Evaluate and benchmark transformer-based surrogate models on non-linear solid mechanics.
- **Milestone 1:** Set up and run the SMART model (Scalable Mesh Autoregressive Transformer) on the Geo-FNO plasticity dataset using local/cluster infrastructure.
- **Milestone 2:** Evaluate the uPDEt (Universal PDE Transformer) architecture on the same test case.

## General AI Behavior & Instructions (Strict)
When generating text or code for this repository, you MUST adhere to the following rules:
- **No Placeholders:** You are strictly forbidden from using placeholders such as `# ... original code here` or any variation.
- **Full Scope Delivery:** Every code block must be complete and syntactically correct within the requested scope.
- **Copy-Paste Readiness:** All code must be production-ready. The user must be able to replace existing blocks with your output without filling in gaps.
- **No Omissions for Brevity:** Never prioritize brevity over completeness. Write logic out in full regardless of length.
- **English Only:** Every comment in the generated code must be in English.
- **Scientific Communication:** No analogies in scientific or mathematical explanations. Provide direct, practical answers that cover 95% of standard scenarios upfront; relegate strict theoretical edge cases to a brief note at the end.

## Python Coding Style

### Naming Conventions
Follow PEP 8 strictly:
- **Classes:** `PascalCase` (e.g., `TransformerSurrogate`, `DatasetLoader`)
- **Functions & Methods:** `snake_case` (e.g., `compute_loss()`, `train_step()`)
- **Variables & Attributes:** `snake_case` (e.g., `learning_rate`, `input_tensor`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `DEFAULT_SEED`, `MAX_EPOCHS`)
- **Private members:** Prefix with a single underscore (e.g., `_build_model()`)

### Formatting & Linting
- **Formatter:** Black or Ruff (line length limit of 88 characters).
- **Linter:** Flake8 or Ruff.
- **Import Sorting:** Group imports with empty lines between groups: (1) Standard library, (2) Third-party (e.g., `numpy`, `torch`), (3) Local application imports.

### Type Hinting
All new functions, variables, parameters, and returns MUST include type hints to facilitate static analysis.
- Use `numpy.ndarray` or `torch.Tensor` specifically for multidimensional arrays.
- For complex returns, use `typing` containers (`List`, `Dict`, `Tuple`, `Optional`), `TypedDict`, or dataclasses.

### Docstrings
Use the **NumPy Docstring Style** for every class and public function. Document all parameters, return types (with array shapes if applicable), and raised exceptions.

### Best Practices & Architecture
- **Pure Functions:** Write pure functions with no side effects wherever possible for parallel execution and testing.
- **Avoid Globals:** Global variables are highly discouraged. Wrap state inside classes or configuration objects (e.g., YAML configs).
- **Resource Cleanup:** Use context managers (`with` statements) for I/O and hardware resource allocation.
- **Modular Design:** Keep functions focused on a single responsibility; decompose large functions into reusable utilities.

## Specific Components Context
- `external/`: Contains third-party Git submodules (e.g., `external/smart`). Do not modify code inside these directories directly; treat them as dependencies.
- `src/`: Core logic library. Contains reusable modules for data loaders, metrics, and model adapters.
- `configs/`: YAML configuration files controlling experimental setups.
- `scripts/`: Executable entry points (e.g., `train.py`, `run_milestone1.sh`) designed for deployment on local machines or the Slurm cluster.