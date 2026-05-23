import argparse
import os
import csv

import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "train": "#A8D8EA",  # pastel blue
    "val": "#FCBAD3",    # pastel pink
    "test": "#B5EAD7",   # pastel mint
}


def load_test_acc_from_csv(csv_path):
    test_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            exp_name = r["exp_name"]
            test_map[exp_name] = float(r["clean"])
    return test_map


def load_train_val_from_npz(exp_name):
    npz_path = os.path.join("figures", f"{exp_name}_metrics.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Metrics file not found: {npz_path}")
    d = np.load(npz_path)
    train_acc = float(d["train_acc_epoch"][-1])
    val_acc = float(d["validation_acc_epoch"][-1])
    return train_acc, val_acc


def collect_rows(exp_names, test_map):
    rows = []
    for exp in exp_names:
        tr, va = load_train_val_from_npz(exp)
        te = test_map[exp]
        rows.append({"exp_name": exp, "train": tr, "val": va, "test": te})
    return rows


def auto_ylim(values, low_clip=0.0, high_clip=1.0):
    vmin = min(values)
    vmax = max(values)
    span = max(vmax - vmin, 1e-6)
    low = max(low_clip, vmin - 0.08 * span)
    high = min(high_clip, vmax + 0.08 * span)
    return low, high


def plot_grouped_bars(rows, title, save_path):
    names = [r["exp_name"] for r in rows]
    x = np.arange(len(rows))
    w = 0.24

    train_vals = [r["train"] for r in rows]
    val_vals = [r["val"] for r in rows]
    test_vals = [r["test"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#FFFDF8")
    ax.set_facecolor("#FFFDF8")

    ax.bar(x - w, train_vals, width=w, color=PALETTE["train"], edgecolor="#666666", linewidth=0.5, label="Train")
    ax.bar(x, val_vals, width=w, color=PALETTE["val"], edgecolor="#666666", linewidth=0.5, label="Validation")
    ax.bar(x + w, test_vals, width=w, color=PALETTE["test"], edgecolor="#666666", linewidth=0.5, label="Test")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    y0, y1 = auto_ylim(train_vals + val_vals + test_vals)
    ax.set_ylim(y0, y1)
    ax.grid(axis="y", linestyle="--", alpha=0.2, color="#9AA0A6")
    ax.legend()

    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")
        spine.set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=r"figures\perturbation_results.csv")
    args = parser.parse_args()

    test_map = load_test_acc_from_csv(args.csv)

    # Figure 1: 3 non-aug models
    no_aug = ["mlp_aug0", "cnn_aug0", "mlp_cnn_aug0"]
    rows_no_aug = collect_rows(no_aug, test_map)
    plot_grouped_bars(
        rows_no_aug,
        "Train / Validation / Test Accuracy (No Aug)",
        r"figures\acc_compare_no_aug_3models.png",
    )

    # Figure 2a: MLP before/after augmentation
    mlp_compare = ["mlp_aug0", "mlp_aug1"]
    rows_mlp = collect_rows(mlp_compare, test_map)
    plot_grouped_bars(
        rows_mlp,
        "Train / Validation / Test Accuracy (MLP: Before vs After Aug)",
        r"figures\acc_compare_mlp_aug_before_after.png",
    )

    # Figure 2b: MLPCNN before/after augmentation
    mlpcnn_compare = ["mlp_cnn_aug0", "mlp_cnn_aug1"]
    rows_mlpcnn = collect_rows(mlpcnn_compare, test_map)
    plot_grouped_bars(
        rows_mlpcnn,
        "Train / Validation / Test Accuracy (MLPCNN: Before vs After Aug)",
        r"figures\acc_compare_mlpcnn_aug_before_after.png",
    )

    print("Saved:")
    print(r"figures\acc_compare_no_aug_3models.png")
    print(r"figures\acc_compare_mlp_aug_before_after.png")
    print(r"figures\acc_compare_mlpcnn_aug_before_after.png")


if __name__ == "__main__":
    main()
