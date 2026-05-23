import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        self.train_scores_epoch = []
        self.dev_scores_epoch = []
        self.train_loss_epoch = []
        self.dev_loss_epoch = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", None)
        save_dir = kwargs.get("save_dir", "best_model")
        log_file = kwargs.get("log_file", None)
        early_stopping = kwargs.get("early_stopping", False)
        es_patience = kwargs.get("es_patience", 5)
        es_min_delta = kwargs.get("es_min_delta", 0.0)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if log_file is None:
            log_file = os.path.join(save_dir, "train.log")

        def log(msg):
            print(msg)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

        with open(log_file, "w", encoding="utf-8") as f:
            f.write("=== Training Log ===\n")

        best_score = 0
        no_improve_count = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            total_iters = int(X.shape[0] / self.batch_size) + 1
            pbar = tqdm(range(total_iters), desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
            epoch_train_losses = []
            epoch_train_scores = []
            for iteration in pbar:
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]
                if train_X.shape[0] == 0:
                    continue

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                epoch_train_losses.append(trn_loss)
                
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)
                epoch_train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                
                if log_iters is not None and log_iters > 0 and (iteration) % log_iters == 0:
                    dev_score, dev_loss = self.evaluate(dev_set)
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)
                    log(f"epoch: {epoch}, iteration: {iteration}")
                    log(f"[Train] loss: {trn_loss}, accuracy: {trn_score}")
                    log(f"[Validation] loss: {dev_loss}, accuracy: {dev_score}")
                    pbar.set_postfix(
                        tr_loss=f"{trn_loss:.4f}",
                        tr_acc=f"{trn_score:.4f}",
                        val_loss=f"{dev_loss:.4f}",
                        val_acc=f"{dev_score:.4f}",
                        lr=f"{self.optimizer.init_lr:.6f}",
                    )

            val_score_epoch, val_loss_epoch = self.evaluate(dev_set)
            self.dev_scores_epoch.append(float(val_score_epoch))
            self.dev_loss_epoch.append(float(val_loss_epoch))

            if len(epoch_train_losses) > 0:
                self.train_loss_epoch.append(float(np.mean(epoch_train_losses)))
                self.train_scores_epoch.append(float(np.mean(epoch_train_scores)))
                log(
                    f"[Epoch {epoch+1}] train_loss: {self.train_loss_epoch[-1]:.6f}, "
                    f"train_acc: {self.train_scores_epoch[-1]:.6f}, "
                    f"validation_loss: {self.dev_loss_epoch[-1]:.6f}, "
                    f"validation_acc: {self.dev_scores_epoch[-1]:.6f}"
                )

            if val_score_epoch > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                log(f"best validation accuracy updated: {best_score:.5f} --> {val_score_epoch:.5f}")
                best_score = val_score_epoch
                no_improve_count = 0
            else:
                if val_score_epoch < best_score + es_min_delta:
                    no_improve_count += 1

            if early_stopping and no_improve_count >= es_patience:
                log(
                    f"early stopping triggered at epoch {epoch+1}, "
                    f"best validation accuracy: {best_score:.5f}, "
                    f"patience: {es_patience}, min_delta: {es_min_delta}"
                )
                break
        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss
    
    def save_model(self, save_path):
        self.model.save_model(save_path)
