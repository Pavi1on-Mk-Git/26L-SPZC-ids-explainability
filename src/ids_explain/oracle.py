import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .config import DataConfig, OracleConfig
from .preprocessing import ProcessedData


class OracleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_classes: int,
        dropout: float,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class OracleModule(LightningModule):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        class_weights: Tensor,
        cfg: OracleConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.model = OracleMLP(input_dim, cfg.hidden_dim, cfg.n_layers, n_classes, cfg.dropout)
        self.register_buffer("class_weights", class_weights)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x, y = batch
        logits = self(x)
        return self.loss_fn(logits, y), logits

    def training_step(self, batch: tuple[Tensor, Tensor], _: int) -> Tensor:
        loss, _ = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], _: int):
        loss, logits = self._shared_step(batch)
        preds = logits.argmax(dim=1)
        acc = (preds == batch[1]).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)


def _class_weights_tensor(y: np.ndarray) -> Tensor:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return torch.from_numpy(weights).to(torch.float32)


def _make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y).to(torch.int64),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )


def train_oracle(
    data: ProcessedData,
    data_cfg: DataConfig,
    oracle_cfg: OracleConfig,
) -> OracleMLP:
    seed_everything(data_cfg.random_seed)

    monitor = oracle_cfg.early_stopping_monitor
    use_validation = monitor.startswith("val")

    if use_validation:
        X_train, X_val, y_train, y_val = train_test_split(
            data.X_train_pca,
            data.y_train,
            test_size=oracle_cfg.val_size,
            random_state=data_cfg.random_seed,
            stratify=data.y_train,
        )
        val_loader = _make_dataloader(X_val, y_val, oracle_cfg.batch_size, shuffle=False)
    else:
        X_train, y_train = data.X_train_pca, data.y_train
        val_loader = None

    train_loader = _make_dataloader(X_train, y_train, oracle_cfg.batch_size, shuffle=True)

    module = OracleModule(
        input_dim=X_train.shape[1],
        n_classes=len(np.unique(y_train)),
        class_weights=_class_weights_tensor(y_train),
        cfg=oracle_cfg,
    )

    mode = "max" if "acc" in monitor else "min"
    early_stop = EarlyStopping(
        monitor=monitor,
        patience=oracle_cfg.early_stopping_patience,
        mode=mode,
        check_on_train_epoch_end=not use_validation,
    )
    checkpoint = ModelCheckpoint(
        dirpath=data_cfg.processed_data_dir,
        filename=oracle_cfg.best_ckpt_name,
        monitor=monitor,
        mode=mode,
        save_weights_only=True,
        save_on_train_epoch_end=not use_validation,
    )
    trainer = Trainer(
        max_epochs=oracle_cfg.max_epochs,
        callbacks=[early_stop, checkpoint],
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=False,
    )
    trainer.fit(module, train_loader, val_loader)

    best_weights = torch.load(
        checkpoint.best_model_path,
        map_location="cpu",
        weights_only=True,
    )
    mlp_state = {k.removeprefix("model."): v for k, v in best_weights["state_dict"].items() if k.startswith("model.")}
    module.model.load_state_dict(mlp_state)
    module.model.eval()
    return module.model


def save_oracle(module: OracleMLP, data_cfg: DataConfig, oracle_cfg: OracleConfig) -> Path:
    path = data_cfg.processed_data_dir / oracle_cfg.model_name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), path)
    return path


def load_oracle(
    data_cfg: DataConfig,
    oracle_cfg: OracleConfig,
) -> OracleMLP:
    path = data_cfg.processed_data_dir / oracle_cfg.model_name
    module = OracleMLP(
        data_cfg.pca_components,
        oracle_cfg.hidden_dim,
        oracle_cfg.n_layers,
        len(data_cfg.classes_to_keep),
        oracle_cfg.dropout,
    )
    module.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    module.eval()
    return module


@torch.inference_mode()
def predict(module: OracleMLP, X: np.ndarray, batch_size: int, device: torch.device = "cpu") -> np.ndarray:
    module.eval()
    module = module.to(device)
    loader = _make_dataloader(X, np.zeros(len(X), dtype=np.int64), batch_size, shuffle=False)
    preds = []
    for x, _ in loader:
        preds.append(module(x.to(device)).argmax(dim=1).cpu())
    return torch.cat(preds).numpy()
