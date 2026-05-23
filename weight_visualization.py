import argparse
import gzip
import os
from struct import unpack

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


def load_test_data():
    test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

    with gzip.open(test_images_path, 'rb') as f:
        _, num, _, _ = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28, 28)

    with gzip.open(test_labels_path, 'rb') as f:
        _, _ = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    return test_imgs.astype(np.float32) / 255.0, test_labs


def build_model(model_type):
    if model_type == 'mlp':
        return nn.models.Model_MLP([28 * 28, 600, 10], 'ReLU')
    if model_type == 'cnn':
        return nn.models.Model_CNN()
    if model_type == 'mlp_cnn':
        return nn.models.Model_MLPCNN()
    raise ValueError(f'Unsupported model_type: {model_type}')


def to_model_input(images, model_type):
    if model_type == 'mlp':
        return images.reshape(images.shape[0], -1)
    return images[:, None, :, :]


def confusion_matrix(preds, labels, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_misclassified_examples(images, labels, preds, save_path, max_show=25):
    wrong_idx = np.where(preds != labels)[0]
    num_show = min(max_show, wrong_idx.shape[0])
    if num_show == 0:
        return
    side = int(np.ceil(np.sqrt(max_show)))
    fig, axes = plt.subplots(side, side, figsize=(10, 10))
    axes = axes.reshape(-1)
    for i in range(side * side):
        axes[i].axis('off')
        if i < num_show:
            idx = wrong_idx[i]
            axes[i].imshow(images[idx], cmap='gray')
            axes[i].set_title(f'T:{labels[idx]} P:{preds[idx]}', fontsize=8)
    fig.suptitle('Misclassified Examples')
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_mlp_weights(model, save_path, max_show=64):
    first_linear = None
    for layer in model.layers:
        if isinstance(layer, nn.op.Linear):
            first_linear = layer
            break
    if first_linear is None:
        return
    w = first_linear.params['W']  # [784, hidden]
    num_show = min(max_show, w.shape[1])
    side = int(np.ceil(np.sqrt(num_show)))
    fig, axes = plt.subplots(side, side, figsize=(10, 10))
    axes = axes.reshape(-1)
    for i in range(side * side):
        axes[i].axis('off')
        if i < num_show:
            axes[i].imshow(w[:, i].reshape(28, 28), cmap='seismic')
            axes[i].set_title(f'#{i}', fontsize=7)
    fig.suptitle('First-Layer MLP Weights')
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_conv_kernels(model, save_path):
    first_conv = None
    for layer in model.layers:
        if isinstance(layer, nn.op.conv2D):
            first_conv = layer
            break
    if first_conv is None:
        return
    w = first_conv.params['W']  # [out, in, k, k]
    out_channels = w.shape[0]
    show = min(32, out_channels)
    side = int(np.ceil(np.sqrt(show)))
    fig, axes = plt.subplots(side, side, figsize=(10, 10))
    axes = axes.reshape(-1)
    for i in range(side * side):
        axes[i].axis('off')
        if i < show:
            axes[i].imshow(w[i, 0], cmap='seismic')
            axes[i].set_title(f'k{i}', fontsize=7)
    fig.suptitle('First Conv Kernels (channel 0)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['mlp', 'cnn', 'mlp_cnn'])
    parser.add_argument('--ckpt', type=str, required=True, help='path to best_model.pickle')
    parser.add_argument('--out_dir', type=str, default='./figures')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = build_model(args.model_type)
    model.load_model(args.ckpt)

    test_imgs, test_labs = load_test_data()
    logits = model(to_model_input(test_imgs, args.model_type))
    preds = np.argmax(logits, axis=1)
    test_acc = (preds == test_labs).mean()

    prefix = f'{args.model_type}_analysis'
    plot_confusion_matrix(
        confusion_matrix(preds, test_labs, num_classes=10),
        os.path.join(args.out_dir, f'{prefix}_confusion_matrix.png'),
    )
    plot_misclassified_examples(
        test_imgs,
        test_labs,
        preds,
        os.path.join(args.out_dir, f'{prefix}_misclassified.png'),
        max_show=25,
    )

    if args.model_type == 'mlp':
        plot_mlp_weights(model, os.path.join(args.out_dir, f'{prefix}_weights.png'))
    elif args.model_type == 'cnn':
        plot_conv_kernels(model, os.path.join(args.out_dir, f'{prefix}_kernels.png'))
    else:
        plot_conv_kernels(model, os.path.join(args.out_dir, f'{prefix}_kernels.png'))
        plot_mlp_weights(model, os.path.join(args.out_dir, f'{prefix}_weights.png'))

    print(f'test_acc={test_acc:.6f}')
    print(f'figures saved to: {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
