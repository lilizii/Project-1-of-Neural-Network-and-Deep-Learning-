# Project-1-of-Neural-Network-and-Deep-Learning-

## 1. Project Overview

This project trains and evaluates three models on MNIST:

- `MLP`
- `CNN`
- `CNN+MLP` (code name: `mlp_cnn`)

It supports:

- baseline training and augmented training
- clean + perturbed evaluation (rotation / translation / Gaussian noise)
- confusion matrix and misclassified sample visualization
- learning-rate sweep (MLP no augmentation)

---

## 2. Environment and Data

- Python 3.x
- Required packages: `numpy`, `matplotlib`, `tqdm`
- MNIST files should exist in:
  - `dataset/MNIST/train-images-idx3-ubyte.gz`
  - `dataset/MNIST/train-labels-idx1-ubyte.gz`
  - `dataset/MNIST/t10k-images-idx3-ubyte.gz`
  - `dataset/MNIST/t10k-labels-idx1-ubyte.gz`

---

## 3. Model Training

Main training entry:

```bash
python test_train.py
```

Common commands:

- Train all models with both no-aug and aug:

```bash
python test_train.py --model all --aug -1
```

- Train all models without augmentation:

```bash
python test_train.py --model all --aug 0
```

- Train single model (example: CNN with augmentation):

```bash
python test_train.py --model cnn --aug 1
```

Useful args:

- `--model`: `all | mlp | cnn | mlp_cnn`
- `--aug`: `-1` (both), `0` (no aug), `1` (aug)
- `--rotate_max_deg`
- `--translate_max_pix`
- `--mix_aug`

Training outputs:

- checkpoints: `best_models/<exp_name>/best_model.pickle`
- logs: `best_models/<exp_name>/train.log`
- curves: `figures/<exp_name>_curves.png`
- metrics: `figures/<exp_name>_metrics.npz`

---

## 4. Model Testing

### 4.1 Basic test script

`test_model.py` can load a saved model and report test accuracy.

```bash
python test_model.py
```

If needed, edit model path inside `test_model.py`.

### 4.2 Perturbation evaluation (recommended)

Evaluate existing trained models under perturbations and generate plots:

```bash
python perturbation_analysis.py
```

Outputs:

- `figures/perturbation_results.csv`
- `figures/perturbation_absolute_accuracy.png`
- `figures/perturbation_accuracy_drop.png`

### 4.3 Learning-rate sweep (MLP no augmentation)

```bash
python lr_sweep_mlp_no_aug.py --lrs "0.001,0.005,0.01,0.05,0.1"
```

Outputs:

- `figures/mlp_no_aug_lr_sweep.csv`
- `figures/mlp_no_aug_lr_sweep_acc_compare.png`
- `figures/mlp_no_aug_lr_sweep_training_curves.png`

---


## 5. Visualization and Analysis

### 5.1 Confusion matrix + misclassified cases

Clean set:

```bash
python error_analysis.py --mode clean
```

Limit models:

```bash
python error_analysis.py --mode clean --targets "mlp_aug0,cnn_aug0,mlp_cnn_aug0"
```

### 5.2 Accuracy comparison plots

Generate train/validation/test comparison plots:

```bash
python plot_split_accuracy_comparison.py
```

Outputs:

- `figures/acc_compare_no_aug_3models.png`
- `figures/acc_compare_mlp_aug_before_after.png`
- `figures/acc_compare_mlpcnn_aug_before_after.png`

