# Spurious Correlations Data Selection

This repository contains code for selecting well-specified OOD sets. This has a few applications
1. Identifying well-specified OOD sets for evaluation domain generalization algorithms.
2. Identifying spurious correlations that harm performance.
3. Identifying samples in data that are most impacted by spurious correaltions.

This repository contains code for few steps:
1. We need to train models; we can launch a job to train many models.
2. We need to take those train models and use their performance on ID/(maybe)OOD examples to select an OOD set.
3. We need to generate natural language descriptions of the difference between the ID and selected OOD set.


## 1. Training Scripts
Generate models.

### 1.1. Direct Training with `train.py`

The `train.py` script allows you to train a single model with specific parameters:

```bash
python train.py \
    --data_dir /path/to/data \
    --output_dir /path/to/output \
    --dataset TerraIncognita \
    --algorithm ERM \
    --model_arch resnet50 \
    --test_envs 0 \
    --seed 0 \
    --steps 5000 \
    --checkpoint_freq 100 \
    --holdout_fraction 0.2 \
    --log_backend wandb
```

Key arguments:
- `--data_dir`: Directory containing the dataset
- `--output_dir`: Directory to save model outputs
- `--dataset`: Dataset name (e.g., TerraIncognita, PACS, WILDSCamelyon)
- `--algorithm`: Training algorithm (e.g., ERM)
- `--model_arch`: Model architecture (e.g., resnet50)
- `--test_envs`: Environment indices to use for testing
- `--log_backend`: Logging backend (wandb, tensorboard, csv, or none)

### 1.2. Batch Training with `main.py`

The `main.py` script allows you to launch multiple training jobs with different configurations; these experiment are i ./experiments.py:

```bash
python main.py launch \
    --experiment PACS_ERM_Transfer \
    --command_launcher local \
    --log_backend wandb
```

Key arguments:
- `command`: Action to perform (launch, delete_incomplete, delete_all)
- `--experiment`: Experiment configuration to use
- `--command_launcher`: How to launch jobs (local or slurm)
- `--log_backend`: Logging backend for all jobs

## 2. Testing Scripts
Select OOD sets

### 2.1. Testing with `test.py`

The `test.py` script allows you to evaluate models to select OOD sets; we also generate figures summarizing the selections:

```bash
python test.py \
    --dataset TerraIncognita \
    --results_dir /path/to/results \
    --loss_type r \
    --num_epochs 3000 \
    --num_trials 5 \
    --train_idxs 0-1-2 \
    --test_idx 3 \
    --output_dir ./results \
    --wandb_project test_spurious_correlations_data_selection
```

### 2.2. Generating Figures with `generate_figures.py`

The `generate_figures.py` script creates visualization plots:

```bash
python generate_figures.py \
    --dataset TerraIncognita \
    --num_domains 4 \
    --train_idxs 0-1-2 \
    --test_idx 3 \
    --results_dir /path/to/results \
    --metric r \
    --training_ood_samples 200 \
    --min_samples 10 \
    --max_samples 800 \
    --output_dir ./figures \
    --explore_selection \
    --selection_threshold 0.5
```

## 3. Compare ID and OOD Samples with Natural Language

### 3.1. Comparing ID and OOD Samples with `compare_ID_OOD_samples.py`

The `compare_ID_OOD_samples.py` script generates natural language descriptions of differences between ID and OOD sets:

```bash
python compare_ID_OOD_samples.py \
    --dataset TerraIncognita \
    --train_envs 0 1 2 \
    --test_envs 3 \
    --selection_path /path/to/selection/vector \
    --num_ID_samples 200 \
    --num_OOD_samples 200 \
    --num_difference_captions 25 \
    --label_idx 0
```

Key arguments:
- `--dataset`: Dataset name
- `--train_envs`: List of environment indices used for training (ID)
- `--test_envs`: List of environment indices used for testing (OOD)
- `--selection_path`: Path to the selection vector file
- `--num_ID_samples`: Number of in-distribution samples to analyze
- `--num_OOD_samples`: Number of out-of-distribution samples to analyze
- `--num_difference_captions`: Number of difference captions to generate
- `--label_idx`: Specific label to analyze (optional) -- when not specified, labels are ignored for analysis

This script:
1. Creates splits between ID and OOD samples
2. Generates captions for both sets
3. Generates difference captions highlighting key distinctions
4. Computes similarity deltas between ID and OOD samples

## 4. Experiment Configurations

The repository includes several predefined experiment configurations:

1. `PACS_ERM_Transfer`: Training on PACS dataset with ERM algorithm
2. `WILDSCamelyon_ERM_Transfer`: Training on WILDSCamelyon dataset with ERM algorithm

Each configuration specifies:
- Dataset and algorithm settings
- Training parameters
- Number of trials
- Loss type
- Number of OOD samples

## 5. Logging

The training process supports multiple logging backends:
- Weights & Biases (wandb)
- TensorBoard
- CSV
- None (console only)

## 6. Requirements

- Python 3.8+
- PyTorch
- PyTorch Lightning
- Weights & Biases (optional)
- TensorBoard (optional)
- Other dependencies listed in requirements.txt

