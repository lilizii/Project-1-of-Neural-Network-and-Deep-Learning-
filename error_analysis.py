import argparse
import gzip
import os
from struct import unpack

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


def load_mnist_test():
    test_images_path = r".\dataset\MNIST\t10k-images-idx3-ubyte.gz"
    test_labels_path = r".\dataset\MNIST\t10k-labels-idx1-ubyte.gz"

    with gzip.open(test_images_path, "rb") as f:
        _, num_test, _, _ = unpack(">4I", f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_test, 28, 28)
    with gzip.open(test_labels_path, "rb") as f:
        _, _ = unpack(">2I", f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    return test_imgs.astype(np.float32) / 255.0, test_labs


def rotate_batch(images, degree=10):
    rad = np.deg2rad(degree)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    n, h, w = images.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    out = np.zeros_like(images)
    for y in range(h):
        for x in range(w):
            xt = x - cx
            yt = y - cy
            src_x = cos_a * xt + sin_a * yt + cx
            src_y = -sin_a * xt + cos_a * yt + cy
            src_xi = int(np.rint(src_x))
            src_yi = int(np.rint(src_y))
            if 0 <= src_xi < w and 0 <= src_yi < h:
                out[:, y, x] = images[:, src_yi, src_xi]
    return out


def translate_batch(images, dx=0, dy=0):
    out = np.zeros_like(images)
    h, w = images.shape[1], images.shape[2]
    xs = max(0, dx)
    xe = min(w, w + dx)
    ys = max(0, dy)
    ye = min(h, h + dy)
    src_xs = max(0, -dx)
    src_xe = src_xs + (xe - xs)
    src_ys = max(0, -dy)
    src_ye = src_ys + (ye - ys)
    out[:, ys:ye, xs:xe] = images[:, src_ys:src_ye, src_xs:src_xe]
    return out


def add_noise(images, std=0.1):
    return np.clip(images + np.random.normal(0.0, std, size=images.shape), 0.0, 1.0)


def to_model_input(images, model_name):
    if model_name == "mlp":
        return images.reshape(images.shape[0], -1)
    return images[:, None, :, :]


def make_model(model_name):
    if model_name == "mlp":
        return nn.models.Model_MLP([28 * 28, 600, 256, 10], "ReLU", [1e-4, 1e-4, 1e-4])
    if model_name == "cnn":
        return nn.models.Model_CNN()
    if model_name == "mlp_cnn":
        return nn.models.Model_MLPCNN()
    raise ValueError(f"Unsupported model: {model_name}")


def get_predictions(model, x):
    logits = model(x)
    return np.argmax(logits, axis=1)


def confusion_matrix_np(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))

    vmax = cm.max() if cm.max() > 0 else 1
    thresh = vmax * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_misclassified(images, y_true, y_pred, title, save_path, max_show=25):
    wrong_idx = np.where(y_true != y_pred)[0]
    if wrong_idx.shape[0] == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No misclassified samples.", ha="center", va="center", fontsize=14)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(save_path, dpi=180)
        plt.close(fig)
        return

    n_show = min(max_show, wrong_idx.shape[0])
    show_idx = wrong_idx[:n_show]
    cols = 5
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")
        if i >= n_show:
            continue
        idx = show_idx[i]
        ax.imshow(images[idx], cmap="gray")
        ax.set_title(f"t:{y_true[idx]} p:{y_pred[idx]}", fontsize=9, color="crimson")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def apply_perturbation(images, mode, rot_deg, dx, dy, noise_std):
    if mode == "clean":
        return images
    if mode == "rot":
        return rotate_batch(images, degree=rot_deg)
    if mode == "trans":
        return translate_batch(images, dx=dx, dy=dy)
    if mode == "noise":
        return add_noise(images, std=noise_std)
    raise ValueError(f"Unsupported mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="clean", choices=["clean", "rot", "trans", "noise"])
    parser.add_argument("--rot_deg", type=float, default=10.0)
    parser.add_argument("--dx", type=int, default=2)
    parser.add_argument("--dy", type=int, default=2)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--max_mis", type=int, default=25)
    parser.add_argument("--include_cnn_aug1", action="store_true")
    parser.add_argument(
        "--targets",
        type=str,
        default="default",
        help="Comma-separated experiment names, e.g. mlp_aug0,cnn_aug0. Use 'default' for report set.",
    )
    args = parser.parse_args()

    np.random.seed(309)
    os.makedirs("figures", exist_ok=True)

    test_imgs, test_labs = load_mnist_test()
    eval_imgs = apply_perturbation(test_imgs, args.mode, args.rot_deg, args.dx, args.dy, args.noise_std)

    if args.targets == "default":
        targets = [
            ("mlp", 0),
            ("cnn", 0),
            ("mlp_cnn", 0),
            ("mlp", 1),
            ("mlp_cnn", 1),
        ]
        if args.include_cnn_aug1:
            targets.append(("cnn", 1))
    else:
        targets = []
        for tag in args.targets.split(","):
            tag = tag.strip()
            if not tag:
                continue
            if "_aug" not in tag:
                raise ValueError(f"Invalid target format: {tag}")
            model_name, aug_str = tag.split("_aug")
            targets.append((model_name, int(aug_str)))

    for model_name, aug_flag in targets:
        exp_name = f"{model_name}_aug{int(aug_flag)}"
        model_path = os.path.join("best_models", exp_name, "best_model.pickle")
        if not os.path.exists(model_path):
            print(f"[Skip] {exp_name} checkpoint not found: {model_path}")
            continue

        model = make_model(model_name)
        model.load_model(model_path)

        x = to_model_input(eval_imgs, model_name)
        pred = get_predictions(model, x)
        acc = float((pred == test_labs).mean())
        cm = confusion_matrix_np(test_labs, pred, num_classes=10)

        cm_path = os.path.join("figures", f"{exp_name}_{args.mode}_confusion_matrix.png")
        err_path = os.path.join("figures", f"{exp_name}_{args.mode}_misclassified.png")
        cm_title = f"{exp_name} | mode={args.mode} | acc={acc:.4f}"
        err_title = f"{exp_name} | mode={args.mode} | Misclassified (t=true, p=pred)"

        plot_confusion_matrix(cm, cm_title, cm_path)
        plot_misclassified(eval_imgs, test_labs, pred, err_title, err_path, max_show=args.max_mis)
        print(f"{exp_name:14s} | mode={args.mode:5s} | acc={acc:.4f}")
        print(f"  saved: {cm_path}")
        print(f"  saved: {err_path}")


if __name__ == "__main__":
    main()
