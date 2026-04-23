# DOXIE

DOXIE is a differential oblivious XGBoost based project for secure XGBoost inference under page-level access-pattern leakage. This repository contains our modifications to Secure XGBoost together with the scripts used for preprocessing, training, prediction and experiment automation.

This README is only for running the project. For the original Secure XGBoost environment and API background, see `README.sxgboost.md`.

## Project Layout

The main experiment code is under `do-enhanced/`:

- `build_project.sh`: build and install the Python package
- `process_dataset.py`: preprocess and encrypt datasets
- `train.py`: train models
- `predict.py`: run one prediction setting
- `run_experiment.py`: automate experiment runs

## Build Modes

`do-enhanced/build_project.sh` supports three modes:

- no flag: non-oblivious mode (`NO`)
- `--O`: oblivious Secure XGBoost (`O`)
- `--DO`: DOXIE (`DO`)

## Installation

1. Follow `README.sxgboost.md` to install the base Secure XGBoost dependencies.
2. Install the Python dependencies used by the DOXIE scripts:

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

## Data and Models

To preprocess a dataset:

```sh
cd do-enhanced
python3 process_dataset.py --dataset <dataset> --size <size>
```

To train models:

```sh
cd do-enhanced
python3 train.py --dataset <dataset> --depths <depths> --num-rounds <rounds>
```

## Quick Start

Build DOXIE:

```sh
cd do-enhanced
./build_project.sh --DO
```

Run one prediction setting:

```sh
cd do-enhanced
python3 predict.py \
  --dataset <dataset> \
  --treesnum <num_trees> \
  --depth <depth> \
  --data-size <batch_size> \
  --epsilon <epsilon> \
  --delta <delta> \
  --shuffle-method <shuffle_method>
```

## Running Experiments

Use `do-enhanced/run_experiment.py` to automate repeated experiment runs.

## Notes

- This project is aimed at batched inference workloads.
- The threat model is page-level access observation.
- The codebase inherits the assumptions and environment constraints of Secure XGBoost.
- `SGX_PTE-Attack/` contains related attack-side material and is separate from the main experiment pipeline.
