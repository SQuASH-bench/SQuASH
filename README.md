# SQuASH: Surrogate Quantum Architecture Search Helper

SQuASH is a benchmark for Quantum Architecture Search (QAS) methods.
It leverages surrogate models to approximate key performance metrics of parameterized quantum circuits (PQCs), enabling fast and reproducible comparisons across diverse QAS strategies.

##  Motivation

The development of quantum algorithms increasingly relies on the careful design and optimization of quantum circuits. In particular, parameterized quantum circuits (PQCs), demand circuit architectures that are not only expressive and trainable but also compatible with hardware constraints.

While QAS methods aim to automate this design process, their evaluation remains challenging due to high resource demands and the lack of unified benchmarks.

**SQuASH** addresses this challenge by:

- Providing **surrogate models** to estimate PQC performance metrics.
- Offering a **benchmarking pipeline** for fair comparison across QAS methods.
- Supplying datasets and tools for **rapid prototyping and reproducible research**.

## 📦 What’s Inside

-  Codebase to integrate SQuASH into custom QAS workflows
-  Pretrained surrogate models for fast performance estimation
-  Datasets used to train the surrogates
-  Examples and scripts for benchmarking QAS methods

## 📁 Structure
```
squash/
├── benchmark/                         # Search spaces and evaluation protocols
│   └── benchmark_surrogate_models/    # Pretrained benchmark models
│
├── data/                               # Datasets used for training/testing
│   ├── raw_data/
│   │   └── demo_dataset_ghz_a
│   ├── processed_data/
│   │   └── gcn_processed_data/
│   │   └── rf_processed_data/
│
├── examples/                           # Notebooks for usage, evaluation, training
│   ├── custom_qas.ipynb
│   ├── train_gcn_surrogate_model_from_scratch.ipynb
│   └── evaluate_gcn_surrogate_model.ipynb
│
├── surrogate_models/
│   ├── prepare_data/
│   │   └── gen_dataset.py              # Generate surrogate-ready datasets
│   ├── architectures/                  # Model definitions and train/test scripts
│   ├── trained_models/                 # Saved model checkpoints
│   └── tuning/                         # Hyperparameter tuning scripts
│
├── config/                             # Optional: config files for models, paths, gates
└── requirements.txt                    # Required dependencies  
```
## 🚀  Getting Started
### 1. Install Dependencies

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install required packages
pip install -r requirements.txt
```
### 2. Run an Example

SQUASH offers several example notebooks to help you get started quickly, depending on your use case.

####  Evaluate Your Custom QAS
If you're developing your own Quantum Architecture Search (QAS) method and want to evaluate it using our benchmark:
```bash
examples/3-SQuASH_integration_into_custom_QAS .ipynb
```
#### Train a Surrogate Model from Scratch
Interested in training a new surrogate model on quantum circuit data? This notebook walks you through dataset loading, model setup, training, and validation:
```bash
examples/0-train_gcn_surrogate_model_from_scratch.ipynb
```



## 📂 Dataset Access

To use the full dataset:

1. Visit [Zenodo](https://zenodo.org/records/15230565)
2. Download the archive
3. Save data into `/data/raw_data/`, if you are interested in raw data and  training new surrogate models, or into `/data/processed_data/gcn/` if you want
to reproduce paper results. 

## 🔧  Requirements

- Python 3.x
- PyTorch and torch-geometric
- Qiskit
- NetworkX, Matplotlib
- scikit-learn, NumPy, SciPy
- Optuna
- tqdm

*(Check the `requirements.txt` for a full list of dependencies.)*

## 🤝 How to Contribute

We welcome contributors! To contribute,

- Check out the [Issues](https://github.com/SQuASH-bench/SQuASH/issues) tab for open tasks. If you detected a problem or have difficulties with documentation, create an issue.
- Got feedback or unsure where to start? Send a message anytime — we're happy to help (see the contact information below).
- If you'd like to work on something, leave a comment on the issue or open a pull request.


## 📄 Citation

If you use this code or dataset in your research, please cite:
```
@unpublished{squash25,
title={Benchmarking Quantum Architecture Search with
Surrogate Assistance},
author={Martyniuk, Darya and Jung, Johannes and Barta, Daniel and Adrian Paschke},
note={Manuscript under submission},
year={2025}
}
```

## Contact information
- Darya Martyniuk, **darya.martyniuk@fokus.fraunhofer.de**
- Johannes Jung, **johannes.jung@fokus.fraunhofer.de**
- Daniel Barta, **daniel.barta@fokus.fraunhofer.de**