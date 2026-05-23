import gzip
import os
import argparse
from struct import unpack

import numpy as np
import matplotlib.pyplot as plt

import mynn as nn
from draw_tools.plot import plot


np.random.seed(309)

BASE_SEED = 309
NUM_EPOCHS = 20
BATCH_SIZE = 64
INIT_LR = 0.05
SCHEDULER_MILESTONES =  [6000, 10000, 13000]
SCHEDULER_GAMMA = 0.5
OPTIMIZER_NAME = 'SGD'
EARLY_STOPPING = True
ES_PATIENCE = 5
ES_MIN_DELTA = 1e-4
ROTATE_MAX_DEG = 6.0
TRANSLATE_MAX_PIX = 1
MIX_AUG = False


def load_mnist():
    train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
    train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'
    test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

    with gzip.open(train_images_path, 'rb') as f:
        _, num_train, _, _ = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_train, 28, 28)
    with gzip.open(train_labels_path, 'rb') as f:
        _, _ = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

    with gzip.open(test_images_path, 'rb') as f:
        _, num_test, _, _ = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_test, 28, 28)
    with gzip.open(test_labels_path, 'rb') as f:
        _, _ = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    train_imgs = train_imgs.astype(np.float32) / 255.0
    test_imgs = test_imgs.astype(np.float32) / 255.0
    return train_imgs, train_labs, test_imgs, test_labs


def split_train_valid(images, labels, valid_size=10000):
    idx = np.random.permutation(images.shape[0])
    images = images[idx]
    labels = labels[idx]
    return images[valid_size:], labels[valid_size:], images[:valid_size], labels[:valid_size]


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


def add_noise(images, std=0.1):
    return np.clip(images + np.random.normal(0.0, std, size=images.shape), 0.0, 1.0)


def scale_batch(images, scale=1.0):
    n, h, w = images.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    out = np.zeros_like(images)
    for y in range(h):
        for x in range(w):
            src_x = (x - cx) / scale + cx
            src_y = (y - cy) / scale + cy
            src_xi = int(np.rint(src_x))
            src_yi = int(np.rint(src_y))
            if 0 <= src_xi < w and 0 <= src_yi < h:
                out[:, y, x] = images[:, src_yi, src_xi]
    return out


def augment_train(images, rotate_max_deg=6.0, translate_max_pix=1, mix_aug=False):
    """
    Create one augmented copy for each image:
    - if mix_aug=True: mix rot/trans/scale
    - else: randomly choose rotation OR translation OR scaling per image
    """
    n = images.shape[0]
    scale_min, scale_max = 0.95, 1.05
    if mix_aug:
        deg = np.random.uniform(-rotate_max_deg, rotate_max_deg)
        dx = np.random.randint(-translate_max_pix, translate_max_pix + 1)
        dy = np.random.randint(-translate_max_pix, translate_max_pix + 1)
        sc = np.random.uniform(scale_min, scale_max)
        aug_rot = rotate_batch(images, degree=deg)
        aug_trans = translate_batch(images, dx=dx, dy=dy)
        aug_scale = scale_batch(images, scale=sc)
        return np.clip((aug_rot + aug_trans + aug_scale) / 3.0, 0.0, 1.0)

    aug_images = np.empty_like(images)
    choice = np.random.randint(0, 3, size=n)  # 0: rotate, 1: translate, 2: scale
    rotate_idx = np.where(choice == 0)[0]
    trans_idx = np.where(choice == 1)[0]
    scale_idx = np.where(choice == 2)[0]

    if rotate_idx.shape[0] > 0:
        deg = np.random.uniform(-rotate_max_deg, rotate_max_deg)
        aug_images[rotate_idx] = rotate_batch(images[rotate_idx], degree=deg)
    if trans_idx.shape[0] > 0:
        dx = np.random.randint(-translate_max_pix, translate_max_pix + 1)
        dy = np.random.randint(-translate_max_pix, translate_max_pix + 1)
        aug_images[trans_idx] = translate_batch(images[trans_idx], dx=dx, dy=dy)
    if scale_idx.shape[0] > 0:
        sc = np.random.uniform(scale_min, scale_max)
        aug_images[scale_idx] = scale_batch(images[scale_idx], scale=sc)
    return aug_images


def to_model_input(images, model_name):
    if model_name == 'mlp':
        return images.reshape(images.shape[0], -1)
    return images[:, None, :, :]


def make_model(model_name):
    if model_name == 'mlp':
        return nn.models.Model_MLP([28 * 28, 600, 256, 10], 'ReLU', [1e-4, 1e-4, 1e-4])
    if model_name == 'cnn':
        return nn.models.Model_CNN()
    if model_name == 'mlp_cnn':
        return nn.models.Model_MLPCNN()
    raise ValueError(f'Unsupported model: {model_name}')


def run_single_experiment(
    model_name,
    use_aug,
    train_imgs,
    train_labs,
    valid_imgs,
    valid_labs,
    test_imgs,
    test_labs,
    rotate_max_deg,
    translate_max_pix,
    mix_aug,
):
    # Fix seed per experiment to make comparison reproducible and fair.
    np.random.seed(BASE_SEED + (100 if use_aug else 0))
    model = make_model(model_name)
    if use_aug:
        aug_x = augment_train(
            train_imgs,
            rotate_max_deg=rotate_max_deg,
            translate_max_pix=translate_max_pix,
            mix_aug=mix_aug,
        )
        train_x = np.concatenate([train_imgs, aug_x], axis=0)
        train_y = np.concatenate([train_labs, train_labs], axis=0)
    else:
        train_x = train_imgs
        train_y = train_labs

    tr_x = to_model_input(train_x, model_name)
    va_x = to_model_input(valid_imgs, model_name)
    te_x = to_model_input(test_imgs, model_name)

    optimizer = nn.optimizer.SGD(init_lr=INIT_LR, model=model)
    scheduler = nn.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=SCHEDULER_MILESTONES,
        gamma=SCHEDULER_GAMMA,
    )
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, batch_size=BATCH_SIZE, scheduler=scheduler)
    exp_name = f'{model_name}_aug{int(use_aug)}'
    save_dir = fr'./best_models/{exp_name}'
    runner.train(
        [tr_x, train_y],
        [va_x, valid_labs],
        num_epochs=NUM_EPOCHS,
        log_iters=None,
        save_dir=save_dir,
        early_stopping=EARLY_STOPPING,
        es_patience=ES_PATIENCE,
        es_min_delta=ES_MIN_DELTA,
    )

    clean_acc, _ = runner.evaluate([te_x, test_labs])
    rot_acc, _ = runner.evaluate([to_model_input(rotate_batch(test_imgs, degree=10), model_name), test_labs])
    trans_acc, _ = runner.evaluate([to_model_input(translate_batch(test_imgs, dx=2, dy=2), model_name), test_labs])
    noise_acc, _ = runner.evaluate([to_model_input(add_noise(test_imgs, std=0.1), model_name), test_labs])
    save_learning_curves(runner, exp_name, clean_acc)

    return {
        'model': model_name,
        'aug': use_aug,
        'valid_best': runner.best_score,
        'test_clean': clean_acc,
        'test_rot10': rot_acc,
        'test_trans2': trans_acc,
        'test_noise0.1': noise_acc,
    }


def save_learning_curves(runner, exp_name, test_acc):
    os.makedirs('./figures', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot(runner, axes)
    fig.suptitle(f'{exp_name} | test_acc={test_acc:.4f}')
    fig.tight_layout()
    fig.savefig(f'./figures/{exp_name}_curves.png', dpi=160)
    plt.close(fig)

    np.savez(
        f'./figures/{exp_name}_metrics.npz',
        train_loss_epoch=np.array(runner.train_loss_epoch),
        validation_loss_epoch=np.array(runner.dev_loss_epoch),
        train_acc_epoch=np.array(runner.train_scores_epoch),
        validation_acc_epoch=np.array(runner.dev_scores_epoch),
        train_loss_iter=np.array(runner.train_loss),
        validation_loss_iter=np.array(runner.dev_loss),
        train_acc_iter=np.array(runner.train_scores),
        validation_acc_iter=np.array(runner.dev_scores),
        test_acc=np.array([test_acc]),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all', choices=['all', 'mlp', 'cnn', 'mlp_cnn'])
    parser.add_argument('--aug', type=int, default=-1, choices=[-1, 0, 1], help='-1: both, 0: no aug, 1: aug')
    parser.add_argument('--rotate_max_deg', type=float, default=ROTATE_MAX_DEG)
    parser.add_argument('--translate_max_pix', type=int, default=TRANSLATE_MAX_PIX)
    parser.add_argument('--mix_aug', type=int, default=int(MIX_AUG), choices=[0, 1])
    args = parser.parse_args()

    print('========== Fair Comparison Config ==========')
    print(f'seed base: {BASE_SEED}')
    print(f'epochs: {NUM_EPOCHS}')
    print(f'batch_size: {BATCH_SIZE}')
    print(f'optimizer: {OPTIMIZER_NAME}')
    print(f'init_lr: {INIT_LR}')
    print(f'lr_scheduler: MultiStepLR(milestones={SCHEDULER_MILESTONES}, gamma={SCHEDULER_GAMMA})')
    print(f'early_stopping: {EARLY_STOPPING}')
    print(f'es_patience: {ES_PATIENCE}')
    print(f'es_min_delta: {ES_MIN_DELTA}')
    print(f'rotate_max_deg: {args.rotate_max_deg}')
    print(f'translate_max_pix: {args.translate_max_pix}')
    print(f'mix_aug: {bool(args.mix_aug)}')
    print('===========================================\n')

    train_imgs, train_labs, test_imgs, test_labs = load_mnist()
    train_imgs, train_labs, valid_imgs, valid_labs = split_train_valid(train_imgs, train_labs)

    model_list = ['mlp', 'cnn', 'mlp_cnn'] if args.model == 'all' else [args.model]
    aug_list = [False, True] if args.aug == -1 else [bool(args.aug)]

    all_results = []
    for model_name in model_list:
        for use_aug in aug_list:
            print(f'\n===== model={model_name}, use_aug={use_aug} =====')
            result = run_single_experiment(
                model_name,
                use_aug,
                train_imgs,
                train_labs,
                valid_imgs,
                valid_labs,
                test_imgs,
                test_labs,
                args.rotate_max_deg,
                args.translate_max_pix,
                bool(args.mix_aug),
            )
            all_results.append(result)
            print(result)

    print('\n========== Final Summary ==========')
    for r in all_results:
        print(
            f"{r['model']:8s} | aug={int(r['aug'])} | valid={r['valid_best']:.4f} | "
            f"clean={r['test_clean']:.4f} | rot10={r['test_rot10']:.4f} | "
            f"trans2={r['test_trans2']:.4f} | noise={r['test_noise0.1']:.4f}"
        )
