import matplotlib.pyplot as plt

colors_set = {'Kraftime': ('#E3E37D', '#968A62')}


def plot(runner, axes, set=colors_set['Kraftime']):
    train_color = set[0]
    dev_color = set[1]

    has_epoch_stats = hasattr(runner, 'train_loss_epoch') and len(runner.train_loss_epoch) > 0
    if has_epoch_stats:
        x_axis = [i + 1 for i in range(len(runner.train_loss_epoch))]
        train_loss = runner.train_loss_epoch
        dev_loss = runner.dev_loss_epoch
        train_scores = runner.train_scores_epoch
        dev_scores = runner.dev_scores_epoch
        x_label = "epoch"
    else:
        x_axis = [i for i in range(len(runner.train_scores))]
        train_loss = runner.train_loss
        dev_loss = runner.dev_loss
        train_scores = runner.train_scores
        dev_scores = runner.dev_scores
        x_label = "iteration"

    axes[0].plot(x_axis, train_loss, color=train_color, label="Train loss")
    axes[0].plot(x_axis, dev_loss, color=dev_color, linestyle="--", label="Dev loss")
    axes[0].set_ylabel("loss")
    axes[0].set_xlabel(x_label)
    if x_label == "epoch":
        axes[0].set_xticks(x_axis)
    axes[0].legend(loc='upper right')

    axes[1].plot(x_axis, train_scores, color=train_color, label="Train accuracy")
    axes[1].plot(x_axis, dev_scores, color=dev_color, linestyle="--", label="Dev accuracy")
    axes[1].set_ylabel("accuracy")
    axes[1].set_xlabel(x_label)
    if x_label == "epoch":
        axes[1].set_xticks(x_axis)
    axes[1].legend(loc='lower right')
