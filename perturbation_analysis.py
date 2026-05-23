import argparse
import gzip
import os
from struct import unpack

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


def load_mnist():
    test_images_path = r".\dataset\MNIST\t10k-images-idx3-ubyte.gz"
    test_labels_path = r".\dataset\MNIST\t10k-labels-idx1-ubyte.gz"

    with gzip.open(test_images_path, "rb") as f:
        _, num_test, _, _ = unpack(">4I", f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_test, 28, 28)
    with gzip.open(test_labels_path, "rb") as f:
        _, _ = unpack(">2I", f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    test_imgs = test_imgs.astype(np.float32) / 255.0
    return test_imgs, test_labs


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


def evaluate_accuracy(model, x, y):
    logits = model(x)
    return float(nn.metric.accuracy(logits, y))


def evaluate_one_model(model_name, aug_flag, test_imgs, test_labs, rot_deg, dx, dy, noise_std):
    exp_name = f"{model_name}_aug{int(aug_flag)}"
    model_path = os.path.join("best_models", exp_name, "best_model.pickle")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = make_model(model_name)
    model.load_model(model_path)

    clean_x = to_model_input(test_imgs, model_name)
    rot_x = to_model_input(rotate_batch(test_imgs, degree=rot_deg), model_name)
    trans_x = to_model_input(translate_batch(test_imgs, dx=dx, dy=dy), model_name)
    noise_x = to_model_input(add_noise(test_imgs, std=noise_std), model_name)

    clean_acc = evaluate_accuracy(model, clean_x, test_labs)
    rot_acc = evaluate_accuracy(model, rot_x, test_labs)
    trans_acc = evaluate_accuracy(model, trans_x, test_labs)
    noise_acc = evaluate_accuracy(model, noise_x, test_labs)

    return {
        "exp_name": exp_name,
        "model": model_name,
        "aug": int(aug_flag),
        "clean": clean_acc,
        "rot": rot_acc,
        "trans": trans_acc,
        "noise": noise_acc,
    }


def save_csv(results, out_csv):
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("exp_name,model,aug,clean,rot,trans,noise,drop_rot,drop_trans,drop_noise\n")
        for r in results:
            f.write(
                f"{r['exp_name']},{r['model']},{r['aug']},"
                f"{r['clean']:.6f},{r['rot']:.6f},{r['trans']:.6f},{r['noise']:.6f},"
                f"{(r['clean'] - r['rot']):.6f},{(r['clean'] - r['trans']):.6f},{(r['clean'] - r['noise']):.6f}\n"
            )


def plot_absolute(results, out_png):
    names = [r["exp_name"] for r in results]
    x = np.arange(len(results))
    width = 0.2

    clean = [r["clean"] for r in results]
    rot = [r["rot"] for r in results]
    trans = [r["trans"] for r in results]
    noise = [r["noise"] for r in results]

    plt.figure(figsize=(13, 5))
    plt.bar(x - 1.5 * width, clean, width=width, label="Clean")
    plt.bar(x - 0.5 * width, rot, width=width, label="Rotation")
    plt.bar(x + 0.5 * width, trans, width=width, label="Translation")
    plt.bar(x + 1.5 * width, noise, width=width, label="Gaussian Noise")

    plt.xticks(x, names, rotation=15)
    plt.ylim(0.75, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Perturbation Stability: Absolute Accuracy")
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_drop(results, out_png):
    names = [r["exp_name"] for r in results]
    x = np.arange(len(results))
    width = 0.25

    drop_rot = [r["clean"] - r["rot"] for r in results]
    drop_trans = [r["clean"] - r["trans"] for r in results]
    drop_noise = [r["clean"] - r["noise"] for r in results]

    plt.figure(figsize=(13, 5))
    plt.bar(x - width, drop_rot, width=width, label="Drop vs Rotation")
    plt.bar(x, drop_trans, width=width, label="Drop vs Translation")
    plt.bar(x + width, drop_noise, width=width, label="Drop vs Gaussian Noise")

    plt.xticks(x, names, rotation=15)
    plt.ylabel("Accuracy Drop (Clean - Perturbed)")
    plt.title("Perturbation Stability: Accuracy Drop")
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rot_deg", type=float, default=10.0)
    parser.add_argument("--dx", type=int, default=2)
    parser.add_argument("--dy", type=int, default=2)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument(
        "--include_cnn_aug1",
        action="store_true",
        help="Include cnn_aug1 in visualization. Default follows report requirement without cnn_aug1.",
    )
    args = parser.parse_args()

    np.random.seed(309)
    os.makedirs("figures", exist_ok=True)
    test_imgs, test_labs = load_mnist()

    targets = [
        ("mlp", 0),
        ("cnn", 0),
        ("mlp_cnn", 0),
        ("mlp", 1),
        ("mlp_cnn", 1),
    ]
    if args.include_cnn_aug1:
        targets.append(("cnn", 1))

    results = []
    for model_name, aug_flag in targets:
        r = evaluate_one_model(
            model_name=model_name,
            aug_flag=aug_flag,
            test_imgs=test_imgs,
            test_labs=test_labs,
            rot_deg=args.rot_deg,
            dx=args.dx,
            dy=args.dy,
            noise_std=args.noise_std,
        )
        results.append(r)
        print(
            f"{r['exp_name']:14s} | clean={r['clean']:.4f} | rot={r['rot']:.4f} | "
            f"trans={r['trans']:.4f} | noise={r['noise']:.4f}"
        )

    csv_path = os.path.join("figures", "perturbation_results.csv")
    abs_png = os.path.join("figures", "perturbation_absolute_accuracy.png")
    drop_png = os.path.join("figures", "perturbation_accuracy_drop.png")

    save_csv(results, csv_path)
    plot_absolute(results, abs_png)
    plot_drop(results, drop_png)

    print("\nSaved files:")
    print(csv_path)
    print(abs_png)
    print(drop_png)


if __name__ == "__main__":
    main()
