import argparse
import gzip
import os
from struct import unpack

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


def load_mnist():
    train_images_path = r".\dataset\MNIST\train-images-idx3-ubyte.gz"
    train_labels_path = r".\dataset\MNIST\train-labels-idx1-ubyte.gz"
    test_images_path = r".\dataset\MNIST\t10k-images-idx3-ubyte.gz"
    test_labels_path = r".\dataset\MNIST\t10k-labels-idx1-ubyte.gz"

    with gzip.open(train_images_path, "rb") as f:
        _, num_train, _, _ = unpack(">4I", f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_train, 28, 28)
    with gzip.open(train_labels_path, "rb") as f:
        _, _ = unpack(">2I", f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

    with gzip.open(test_images_path, "rb") as f:
        _, num_test, _, _ = unpack(">4I", f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_test, 28, 28)
    with gzip.open(test_labels_path, "rb") as f:
        _, _ = unpack(">2I", f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    return train_imgs.astype(np.float32) / 255.0, train_labs, test_imgs.astype(np.float32) / 255.0, test_labs


def split_train_valid(images, labels, valid_size=10000, seed=309):
    np.random.seed(seed)
    idx = np.random.permutation(images.shape[0])
    images = images[idx]
    labels = labels[idx]
    return images[valid_size:], labels[valid_size:], images[:valid_size], labels[:valid_size]


def run_one_lr(lr, train_x, train_y, valid_x, valid_y, test_x, test_y, epochs, batch_size, base_seed):
    np.random.seed(base_seed)
    model = nn.models.Model_MLP([28 * 28, 600, 256, 10], "ReLU", [1e-4, 1e-4, 1e-4])
    optimizer = nn.optimizer.SGD(init_lr=lr, model=model)
    scheduler = nn.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=[6000, 10000, 13000],
        gamma=0.5,
    )
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    runner = nn.runner.RunnerM(
        model, optimizer, nn.metric.accuracy, loss_fn, batch_size=batch_size, scheduler=scheduler
    )

    save_dir = os.path.join("best_models", f"mlp_lr_{str(lr).replace('.', 'p')}")
    runner.train(
        [train_x, train_y],
        [valid_x, valid_y],
        num_epochs=epochs,
        log_iters=None,
        save_dir=save_dir,
        early_stopping=True,
        es_patience=5,
        es_min_delta=1e-4,
    )

    test_acc, _ = runner.evaluate([test_x, test_y])
    train_acc = float(runner.train_scores_epoch[-1])
    val_acc = float(runner.dev_scores_epoch[-1])
    return (
        train_acc,
        val_acc,
        float(test_acc),
        list(runner.train_loss_epoch),
        list(runner.dev_scores_epoch),
    )


def plot_results(rows, save_path):
    labels = [f"lr={r['lr']}" for r in rows]
    x = np.arange(len(rows))
    w = 0.24

    train_vals = [r["train_acc"] for r in rows]
    val_vals = [r["val_acc"] for r in rows]
    test_vals = [r["test_acc"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#FFFDF8")
    ax.set_facecolor("#FFFDF8")

    ax.bar(x - w, train_vals, width=w, label="Train", color="#A8D8EA", edgecolor="#666666", linewidth=0.5)
    ax.bar(x, val_vals, width=w, label="Validation", color="#FCBAD3", edgecolor="#666666", linewidth=0.5)
    ax.bar(x + w, test_vals, width=w, label="Test", color="#B5EAD7", edgecolor="#666666", linewidth=0.5)

    all_vals = train_vals + val_vals + test_vals
    vmin, vmax = min(all_vals), max(all_vals)
    span = max(vmax - vmin, 1e-6)
    ax.set_ylim(max(0.0, vmin - 0.08 * span), min(1.0, vmax + 0.08 * span))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Accuracy")
    ax.set_title("MLP (No Aug) with Different Learning Rates")
    ax.grid(axis="y", linestyle="--", alpha=0.2, color="#9AA0A6")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_csv(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("lr,train_acc,val_acc,test_acc\n")
        for r in rows:
            f.write(f"{r['lr']},{r['train_acc']:.6f},{r['val_acc']:.6f},{r['test_acc']:.6f}\n")


def plot_training_curves(rows, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor("#FFFDF8")
    colors = ["#A8D8EA", "#FCBAD3", "#B5EAD7", "#AA96DA", "#FFDAC1", "#C7CEEA"]

    # Left: train loss
    axes[0].set_facecolor("#FFFDF8")
    for i, r in enumerate(rows):
        x = np.arange(1, len(r["train_loss_curve"]) + 1)
        axes[0].plot(
            x,
            r["train_loss_curve"],
            label=f"lr={r['lr']}",
            color=colors[i % len(colors)],
            linewidth=2.0,
        )
    axes[0].set_title("Train Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(axis="both", linestyle="--", alpha=0.2, color="#9AA0A6")
    axes[0].legend()

    # Right: validation accuracy
    axes[1].set_facecolor("#FFFDF8")
    for i, r in enumerate(rows):
        x = np.arange(1, len(r["val_acc_curve"]) + 1)
        axes[1].plot(
            x,
            r["val_acc_curve"],
            label=f"lr={r['lr']}",
            color=colors[i % len(colors)],
            linewidth=2.0,
        )
    axes[1].set_title("Validation Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(axis="both", linestyle="--", alpha=0.2, color="#9AA0A6")
    axes[1].legend()

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color("#BBBBBB")
            spine.set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrs", type=str, default="0.005,0.01,0.05,0.1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=309)
    args = parser.parse_args()

    lr_list = [float(x.strip()) for x in args.lrs.split(",") if x.strip()]

    train_imgs, train_labs, test_imgs, test_labs = load_mnist()
    tr_i, tr_y, va_i, va_y = split_train_valid(train_imgs, train_labs, valid_size=10000, seed=args.seed)

    train_x = tr_i.reshape(tr_i.shape[0], -1)
    valid_x = va_i.reshape(va_i.shape[0], -1)
    test_x = test_imgs.reshape(test_imgs.shape[0], -1)

    rows = []
    for lr in lr_list:
        train_acc, val_acc, test_acc, train_loss_curve, val_acc_curve = run_one_lr(
            lr=lr,
            train_x=train_x,
            train_y=tr_y,
            valid_x=valid_x,
            valid_y=va_y,
            test_x=test_x,
            test_y=test_labs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            base_seed=args.seed,
        )
        rows.append(
            {
                "lr": lr,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "train_loss_curve": train_loss_curve,
                "val_acc_curve": val_acc_curve,
            }
        )
        print(f"lr={lr:<6} | train={train_acc:.4f} | val={val_acc:.4f} | test={test_acc:.4f}")

    os.makedirs("figures", exist_ok=True)
    csv_path = r"figures\mlp_no_aug_lr_sweep.csv"
    fig_path = r"figures\mlp_no_aug_lr_sweep_acc_compare.png"
    curve_path = r"figures\mlp_no_aug_lr_sweep_training_curves.png"
    save_csv(rows, csv_path)
    plot_results(rows, fig_path)
    plot_training_curves(rows, curve_path)

    print("\nSaved:")
    print(csv_path)
    print(fig_path)
    print(curve_path)


if __name__ == "__main__":
    main()
