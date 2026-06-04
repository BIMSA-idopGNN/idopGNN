import torch
import pandas as pd
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def evaluate_loss(model, X, edge_index, edge_attr, y, mask, criterion):
    """
    Evaluate the loss of the model on a specific data split.

    Args:
        model (nn.Module): The model to evaluate.
        X (Tensor): Node feature matrix [N, F].
        edge_index (Tensor): Edge indices [2, E].
        edge_attr (Tensor): Edge attributes [E, 1].
        y (Tensor): True labels [N, C].
        mask (Tensor): Boolean mask for the evaluation subset.
        criterion: Loss function.

    Returns:
        float: The computed loss value.
    """
    model.eval()
    with torch.no_grad():
        A0 = to_dense_adj(edge_index, edge_attr=edge_attr)[0].squeeze(-1)
        out = model(X, A0)
        loss = criterion(out[mask], y[mask])
    return loss.item()


class EarlyStoppingByLoss:
    """
    Early stopping based on validation loss.
    Stops training when validation loss does not improve for 'patience' epochs.
    """
    def __init__(self, patience=None):
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def step(self, current_loss):
        """
        Check whether early stopping should be triggered.

        Args:
            current_loss (float): Current epoch validation loss.

        Returns:
            bool: True if early stopping is triggered, False otherwise.
        """
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_class_early_stopping_loss(model, data, optimizer, criterion,
                                   max_epochs=None, patience=None):
    """
    Train the model with early stopping based on validation loss.

    Args:
        model (nn.Module): Model to train.
        X (Tensor): Node feature matrix [N, F].
        edge_index (Tensor): Edge indices [2, E].
        edge_attr (Tensor): Edge attributes [E, 1].
        y (Tensor): Ground truth labels [N, C].
        train_mask (Tensor): Boolean mask for training samples.
        val_mask (Tensor): Boolean mask for validation samples.
        optimizer: Optimizer for gradient updates.
        criterion: Loss function.
        max_epochs (int): Maximum number of epochs.
        patience (int): Number of epochs to wait for improvement before stopping.

    Returns:
        tuple: (best_val_loss, model)
    """
    early_stopper = EarlyStoppingByLoss(patience=patience)
    best_val_loss = float('inf')
    best_model_state = None

    X = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    y = data.y
    train_mask = data.train_mask
    val_mask = data.val_mask

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()

        A0 = to_dense_adj(edge_index, edge_attr=edge_attr)[0].squeeze(-1)

        out = model(X, A0)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        val_loss = evaluate_loss(model, X, edge_index, edge_attr, y, val_mask, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if early_stopper.step(val_loss):
            print(f"Early stop triggered at epoch {epoch}, best val loss: {best_val_loss:.4f}")
            best_model_state = model.state_dict()
            break


    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_val_loss, model


def evaluate_metrics_class(model, X, edge_index, edge_attr, y, mask):
    """
    Evaluate model performance using multiple metrics (AUROC, AUPRC, F1).

    Args:
        model (nn.Module): Trained model.
        X (Tensor): Node feature matrix [N, F].
        edge_index (Tensor): Edge indices [2, E].
        edge_attr (Tensor): Edge attributes [E, 1].
        y (Tensor): Ground truth labels [N, C].
        mask (Tensor): Boolean mask for evaluation subset.

    Returns:
        dict: Dictionary containing AUROC, AUPRC, micro-F1, macro-F1.
    """
    model.eval()
    with torch.no_grad():
        A0 = to_dense_adj(edge_index, edge_attr=edge_attr)[0].squeeze(-1)

        out = model(X, A0)
        y_true = y[mask].cpu().detach().numpy()
        y_pred = out[mask].cpu().detach().numpy()

        class_auroc = []
        try:
            auroc = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
        except Exception:
            auroc = float('nan')

        try:
            auprc = average_precision_score(y_true, y_pred, average='macro')
        except Exception:
            auprc = float('nan')

        micro_f1 = f1_score(y_true, y_pred > 0.5, average='micro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred > 0.5, average='macro', zero_division=0)

    return {
        'auroc': auroc,
        'auprc': auprc,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'class_auroc': class_auroc
    }
