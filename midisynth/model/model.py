import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import (RNN, Activation, Loss, Optimizer, PoolingMethod,
                    get_activation, get_loss_fn, get_optimizer)


class Baseline(nn.Module):
    def __init__(self, model_cfg: DictConfig) -> None:
        super().__init__()
        # layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # null variables
        self.optimizer = None
        self.use_wandb = False
        self.channel_first = model_cfg.get("channel_first", True)

        # declare weights
        self.conv = nn.ModuleList([
            self._get_conv_block(**conv_params)
            for conv_params
            in model_cfg.get("cnn", [])
        ])
        pooling_cfg = model_cfg.get("pooling", {})
        self.pooling = self._get_pooling(pooling_cfg.get("method", None))
        self.pooling_dropout = nn.Dropout(pooling_cfg.get("dropout", 0.))
        self.rnn = self._get_rnn(
            model_cfg.get("rnn", None)
        )
        self.linear = nn.ModuleList([
            self._get_linear(**params)
            for params in model_cfg.get("linear", [])
        ])

        # send model to device
        self.to(self.device)

    def init_wandb(
        self,
        wandb_apikey: str,
        config: Dict[str, Any],
        project: Optional[str] = None,
        name: Optional[str] = None,
        entity: Optional[str] = None,
    ) -> None:
        # wandb init
        wandb.login(key=wandb_apikey)
        wandb.init(
            project=project,
            name=name,
            entity=entity,
            config=config
        )

        # log all parameters / gradients
        wandb.watch(self, log="all")

    def set_optimizer(
        self,
        optimizer: str,
        learning_rate: float,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> None:
        self.scheduler = scheduler
        optim_cls = get_optimizer(optimizer)
        self.optimizer = optim_cls(self.parameters(), lr=learning_rate)

    def _get_conv_block(
        self, 
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        stride: int,
        padding: str = "same",
        activation: Optional[Activation] = None
    ) -> nn.Sequential:
        block = [
            nn.Conv1d(
                in_channel, 
                out_channel, 
                kernel_size, 
                stride=stride,
                padding=padding
            ),
        ]
        if activation is not None:
            block.append(nn.BatchNorm1d(out_channel))
            block.append(get_activation(activation))
        return nn.Sequential(*block)

    def _get_pooling(self, method: PoolingMethod) -> nn.Module:
        if method is None:
            return None
        
        if isinstance(method, str):
            method = PoolingMethod(method.lower().strip())

        if method.value == PoolingMethod.GAP1D.value:
            def gap1d(x: Tensor) -> Tensor:
                return x.mean(dim=-1)  # torch cnn1d (B, C, T)
            return gap1d
        elif method.value == PoolingMethod.FLATTEN.value:
            def flatten(x: Tensor) -> Tensor:
                batch_size = x.shape[0]
                return x.reshape(batch_size, -1)
            return flatten

        # reshape from (B, C, T) -> (B, T, C)
        # if not prompt any pool choice, 
        # output from conv will be converted
        # to compat rnn
        def reshape(x: Tensor) -> Tensor:
            return x.transpose(1, 2)
        return reshape

    def _get_rnn(self, rnn_config: DictConfig) -> nn.Module:
        if rnn_config is None:
            return None

        rnn_name = RNN(rnn_config.name.lower().strip())
        if rnn_name.value == RNN.LSTM.value:
            return nn.LSTM(**rnn_config.param)
        elif rnn_name.value == RNN.GRU.value:
            return nn.GRU(**rnn_config.param)
        else:
            return None

    def _get_linear(
        self, 
        feat_in: int, 
        feat_out: int,
        activation: Optional[Activation] = None,
    ) -> nn.Module:
        block = [nn.Linear(feat_in, feat_out)]
        if activation is not None:
            block.append(get_activation(activation))
        return nn.Sequential(*block)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.channel_first:
            x = x.transpose(1, 2)  # convert (B, T, C) to (B, C, T)
        for conv in self.conv:
            x = conv(x)
        x = self.pooling(x)
        x = self.pooling_dropout(x)
        if self.rnn is not None:
            x = self.rnn(x)
        for fc in self.linear:
            x = fc(x)
        return x

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 30,
        optimizer: Optimizer = Optimizer.ADAMW,
        learning_rate: float = 3e-4,
        loss_fn: Loss = Loss.XENT,
        mixed_precision: bool = False,
        prog_bar: bool = True,
        checkpoint_dir: str = "weights",
        save_every: int = 1,
        wandb_apikey: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_project: Optional[str] = None,
        wandb_name: Optional[str] = None,
        wandb_entity: Optional[str] = None,
    ) -> Any:
        if wandb_apikey is not None:
            self.init_wandb(
                wandb_apikey=wandb_apikey,
                config=wandb_config,
                project=wandb_project,
                name=wandb_name,
                entity=wandb_entity
            )
            self.use_wandb = True

        # setup loss fn
        loss_fn = get_loss_fn(loss_fn)

        # setup optimizer
        self.set_optimizer(optimizer, learning_rate)

        if mixed_precision:
            scaler = GradScaler()

        val_losses, train_losses = [], []
        val_accs, train_accs = [], []

        for epoch in range(epochs):

            # init training phase
            start_time = time.time()
            self.train()
            train_batch_losses = []
            train_batch_accs = []

            # iterate over training batch
            train_iter = tqdm(
                train_dataloader,
                leave=False,
                desc=f"Training Epoch {epoch+1}/{epochs}",
                unit=" batch"
            ) if prog_bar else train_dataloader
            for idx, (x, y) in enumerate(train_iter):
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                if mixed_precision:
                    # mixed precision training
                    with autocast():
                        y_pred = self(x)
                        loss = loss_fn(y_pred, y)

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    y_pred = self(x)
                    loss = loss_fn(y_pred, y)

                    loss.backward()
                    self.optimizer.step()

                train_batch_loss = loss.detach().cpu()
                train_batch_accuracy = (y_pred.argmax(
                    dim=-1) == y).sum() / x.shape[0]

                train_batch_losses.append(
                    train_batch_loss.detach().cpu().numpy())
                train_batch_accs.append(
                    train_batch_accuracy.detach().cpu().numpy())

                train_losses.append(train_batch_loss.detach().cpu().numpy())
                train_accs.append(train_batch_accuracy.detach().cpu().numpy())

                if prog_bar:
                    train_iter.set_postfix({
                        "Batch": idx + 1,
                        "Train Loss": train_batch_loss.detach().cpu().numpy(),
                        "Train Accuracy": train_batch_accuracy.detach().cpu().numpy()
                    })

                if wandb_apikey is not None:
                    wandb.log({
                        "train_loss": train_batch_loss,
                        "train_acc": train_batch_accuracy,
                        "epoch": epoch+1
                    })

            train_epoch_loss = np.mean(train_batch_losses)
            train_epoch_acc = np.mean(train_batch_accs)
            train_time = time.time() - start_time

            if val_dataloader is not None:
                # calculating validation data
                start_time = time.time()
                self.eval()
                val_batch_losses = []
                val_batch_accs = []
                for x, y in val_dataloader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    with torch.no_grad():
                        y_pred = self(x)
                        loss = loss_fn(y_pred, y)

                        val_batch_accuracy = (y_pred.argmax(
                            dim=-1) == y).sum() / x.shape[0]
                        val_batch_loss = loss.cpu()

                    val_batch_losses.append(
                        val_batch_loss.detach().cpu().numpy())
                    val_batch_accs.append(
                        val_batch_accuracy.detach().cpu().numpy())

                val_epoch_loss = np.mean(val_batch_losses)
                val_epoch_acc = np.mean(val_batch_accs)
                val_time = time.time() - start_time

                val_losses.append(val_epoch_loss)
                val_accs.append(val_epoch_acc)

                if wandb_apikey is not None:
                    wandb.log({
                        "val_loss": val_batch_loss,
                        "val_acc": val_batch_accuracy,
                    })

                # update progbar
                if prog_bar:
                    train_iter.set_postfix({
                        "Train Loss": train_epoch_loss,
                        "Train Accuracy": train_epoch_acc,
                        "Validation Loss": val_epoch_loss,
                        "Validation Accuracy": val_epoch_acc
                    })
                    train_iter.close()

            else:
                # if got no val data
                # update progbar
                if prog_bar:
                    train_iter.set_postfix({
                        "Train Loss": train_epoch_loss,
                        "Train Accuracy": train_epoch_acc,
                    })
                    train_iter.close()

            if (epoch+1) % save_every == 0:
                weight_path = f"{checkpoint_dir}/epoch{epoch+1}.pt"
                os.makedirs(os.path.dirname(weight_path), exist_ok=True)
                logging.info(f"Saving weights at {weight_path}...")
                torch.save(self.state_dict(), weight_path)

            logging.info(f"[EPOCH {epoch+1} / {epochs}]\nTrain Loss: {train_epoch_loss:.6f}\tTrain Accuracy: {train_epoch_acc*100:.2f}%\nValidation Loss: {val_epoch_loss:.6f}\tValidation Accuracy: {val_epoch_acc*100:.2f}%")

        return {
            "train_loss": train_losses,
            "train_acc": train_accs,
            "val_loss": val_losses,
            "val_acc": val_accs
        }

    def evaluate(self, test_dataloader: DataLoader, prog_bar: bool = True) -> Dict[str, Any]:
        logits = []
        labels = []
        self.eval()
        test_iter = tqdm(test_dataloader, desc="Evaluating",
                         unit=" batch") if prog_bar else test_dataloader
        for x, y in test_iter:
            with torch.no_grad():
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = self(x)

                logits.append(y_pred.cpu().numpy())
                labels.append(y.cpu().numpy())

        logits = np.concatenate(logits)
        labels = np.concatenate(labels)

        predictions = logits.argmax(-1)
        accuracy = accuracy_score(labels, predictions)
        macro_recall = recall_score(labels, predictions, average="macro")
        macro_precision = precision_score(labels, predictions, average="macro")
        macro_f1 = f1_score(labels, predictions, average="macro")
        cm = confusion_matrix(labels, predictions)

        if self.use_wandb:
            wandb.run.summary["test_accuracy"] = accuracy
            wandb.run.summary["test_precision"] = macro_precision
            wandb.run.summary["test_recall"] = macro_recall
            wandb.run.summary["test_f1"] = macro_f1
            wandb.log(
                {
                    "test_cm": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=labels,
                        preds=logits.argmax(-1),
                    )
                }
            )

            wandb.finish()

        return {
            "logits": logits,
            "accuracy": accuracy,
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
            "confusion_matrix": cm
        }
