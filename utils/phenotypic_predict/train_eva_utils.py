import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from copy import deepcopy

from torch_geometric.utils import to_dense_adj

def train_regression_early_stopping_loss(model, data, optimizer, criterion,
                                    max_epochs=1000, patience=100, verbose=True):
    """
    Train a model with early stopping based on validation loss.

    Args:
        model: PyTorch model to train.
        optimizer: PyTorch optimizer.
        A0: Initial adjacency matrix or relevant graph prior.
        max_epochs: Maximum number of epochs for training.
        patience: Early stopping patience (number of epochs to wait for improvement).
        verbose: Whether to print progress logs.

    Returns:
        model: Model with the best validation loss loaded.
        best_val_loss: The best validation loss achieved.
    """

    best_val_loss = float('inf')
    wait = 0
    best_model_state = None

    def train_one_epoch():
        model.train()
        total_loss = 0
        for graph in data['train_loader']:
            optimizer.zero_grad()
            loss = criterion(model, graph, graph.y,
                                      lambda_l1_values=0,
                                      A0=data['A0'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data['train_loader'])

    def evaluate(loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for graph in loader:
                loss = criterion(model, graph, graph.y,
                                          lambda_l1_values=0,
                                          A0=data['A0'])
                total_loss += loss.item()
        return total_loss / len(loader)

    for epoch in range(max_epochs):
        train_loss = train_one_epoch()
        val_loss = evaluate(data['val_loader'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}, best val loss: {best_val_loss:.4f}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return  best_val_loss, model


def dirichlet_logpdf(y, alpha, batch_size, eps=1e-8):

    if len(y.shape) == 1:
        y = y.unsqueeze(0)
    if len(alpha.shape) == 1:
        alpha = alpha.unsqueeze(0)

    y = y.view(batch_size, -1)
    log_B = torch.sum(torch.lgamma(alpha), dim=-1) - torch.lgamma(torch.sum(alpha, dim=-1))
    log_pdf = torch.sum((alpha - 1) * torch.log(y + eps), dim=-1) - log_B
    return log_pdf

def neg_log_likelihood(model, graph, target, lambda_l1_values=0.0, A0=None):
    """
    Compute negative log-likelihood for Dirichlet output
    + optional L1 regularization on weights
    """

    output = model(graph.x, A0, graph.batch)
    batch_size = graph.batch.max().item() + 1
    alpha = output

    if len(target.shape) == 1:
        target = target.view(-1, alpha.shape[1])

    nll = -dirichlet_logpdf(target, alpha, batch_size)
    nll = torch.mean(nll)

    # L1 penalty
    l1_penalty = 0
    for conv in model.convs:
        if hasattr(conv, 'lin') and hasattr(conv.lin, 'weight'):
            l1_penalty += torch.sum(torch.abs(conv.lin.weight))
    if hasattr(model, 'final') and hasattr(model.final, 'weight'):
        l1_penalty += torch.sum(torch.abs(model.final.weight))

    l1_penalty *= lambda_l1_values
    total_loss = nll + l1_penalty
    return total_loss


def aitchison_distance(x, y):
    """
    x, y: numpy arrays or tensors (1D or 2D)
    Returns: Euclidean distance in CLR-transformed space
    """
    if hasattr(x, 'numpy'):
        x = x.numpy()
    if hasattr(y, 'numpy'):
        y = y.numpy()

    g_x = np.exp(np.mean(np.log(x), axis=-1, keepdims=True))
    g_y = np.exp(np.mean(np.log(y), axis=-1, keepdims=True))
    clr_x = np.log(x / g_x)
    clr_y = np.log(y / g_y)
    return np.linalg.norm(clr_x - clr_y, axis=-1)


def mean_aitchison_distance_class(X, Y):
    distances = []
    X = X.T
    Y = Y.T
    for x, y in zip(X, Y):
        distances.append(aitchison_distance(x, y))
    return distances

def mean_aitchison_distance_sample(X, Y):
    distances = [aitchison_distance(x, y) for x, y in zip(X, Y)]
    return np.mean(distances)

def r2_aitchison_per_column(y_true, y_pred):
    """
    Compute R² metric for compositional data using Aitchison distance
    Returns: array of R² values per column
    """
    n_columns = y_true.shape[1]
    r2_values = []

    for i in range(n_columns):
        col_true = y_true[:, i].reshape(-1, 1)
        col_pred = y_pred[:, i].reshape(-1, 1)

        # normalize compositional vectors
        col_true /= np.sum(col_true)
        col_pred /= np.sum(col_pred)

        # Residual sum of squares
        res_distances = np.array(mean_aitchison_distance_class(col_true, col_pred))
        ss_res = np.sum(res_distances ** 2)

        # Total sum of squares
        mean_comp = np.mean(col_true, axis=0, keepdims=True)
        mean_mat = np.repeat(mean_comp, col_true.shape[0], axis=0)
        tot_distances = np.array(mean_aitchison_distance_class(col_true, mean_mat))
        ss_tot = np.sum(tot_distances ** 2)

        r2 = 1 - ss_res / ss_tot
        r2_values.append(r2)

    return np.array(r2_values)

def split_indices(data, prefixes=['XD', 'ZD', 'YD', 'ED'], ratios=(0.6, 0.2, 0.2), random_state=3):
    assert abs(sum(ratios) - 1.0) < 1e-6, "warning"
    train_idx, val_idx, test_idx = [], [], []

    for prefix in prefixes:
        prefix_idx = data.index[data.index.str.startswith(prefix)].tolist()
        train_sub, valtest_sub = train_test_split(
            prefix_idx,
            train_size=ratios[0],
            random_state=random_state
        )
        val_ratio = ratios[1] / (ratios[1] + ratios[2])
        val_sub, test_sub = train_test_split(
            valtest_sub,
            train_size=val_ratio,
            random_state=random_state
        )
        train_idx.extend(train_sub)
        val_idx.extend(val_sub)
        test_idx.extend(test_sub)

    return train_idx, val_idx, test_idx

def create_graph_data(sample_data, edge_index, l, embeding, edge_attr, device):

    if isinstance(sample_data, torch.Tensor):
        x = sample_data.clone().detach().view(-1, 1).to(device)
    else:
        if hasattr(sample_data, 'values'):
            sample_data = sample_data.values
        x = torch.tensor(sample_data, dtype=torch.float32).view(-1, 1).to(device)

    if isinstance(embeding, torch.Tensor):
        embeding = embeding.to(device)
    else:
        if hasattr(embeding, 'values'):
            embeding = embeding.values
        embeding = torch.tensor(embeding, dtype=torch.float32).to(device)
    x = x * embeding
    y = l.clone().detach().to(device)
    edge_attr_tensor = edge_attr.to(device) if isinstance(edge_attr, torch.Tensor) else torch.tensor(
        edge_attr.values, dtype=torch.float32).to(device)
    return Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr_tensor)


import numpy as np
import warnings
from sklearn.metrics import r2_score


def evaluate_metrics_regression(model, data):
    """
    Evaluate a trained model on the test set and compute multiple metrics.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for the test set.
        A0: Adjacency matrix or graph prior used in the model.
        seed: Random seed (optional, used for warning messages).
        lr, hidden_dim, layer, lamda, topk, kappa, ealpha, nalpha, dropout: Hyperparameters, used for logging.

    Returns:
        dict: A dictionary containing all computed metrics:
            - r2_test
            - aitchison_class_test
            - aitchison_sample_test
            - r2_aitchison_test
            - r2_class
            - aitchison_class1
            - r2_aitchison_class
    """
    model.eval()
    predictions = []
    A0 = to_dense_adj(data['edge_index'], edge_attr=data['edge_attr'])[0].squeeze(-1)


    with torch.no_grad():
        for graph in data['test_loader']:
            output = model(graph.x, A0, graph.batch)
            y_pred = output / output.sum(dim=1, keepdim=True)
            predictions.append(y_pred.cpu().numpy())

    predictions = np.vstack(predictions)
    y_test_np = np.vstack([data.y.cpu().numpy() for data in data['test_loader'].dataset])

    if np.isnan(y_test_np).any() or np.isnan(predictions).any():
        warnings.warn(f"NaN detected in metrics for seed, skipping this seed")
        return None

    # Only use first 7 columns
    y_test_np1 = y_test_np[:, :7]
    predictions1 = predictions[:, :7]

    r2_test = r2_score(y_test_np1, predictions1)
    aitchisones_class = mean_aitchison_distance_class(y_test_np1, predictions1)
    aitchison_class= np.mean(aitchisones_class)
    r2_aitchisons = r2_aitchison_per_column(y_test_np1, predictions1)
    r2_aitchison = np.mean(r2_aitchisons)

    r2_class = r2_score(y_test_np1, predictions1, multioutput='raw_values')

    r2_aitchison_class = r2_aitchisons

    print(f"Test R2: {r2_test:.4f}, "
          f"Aitchison Distance: {aitchison_class:.4f}, "
          f"R2 Aitchison: {r2_aitchison:.4f}, ")

    if np.isnan(r2_test) or np.isnan(aitchison_class) or np.isnan(r2_aitchison):
        warnings.warn(f"NaN detected in test metrics for seed, skipping this seed")
        return None

    return {
        'r2': r2_test,
        'aitchison': aitchison_class,
        'r2_aitchison': r2_aitchison,
        'r2_class': r2_class,
        'aitchison_class': aitchisones_class,
        'r2_aitchison_class': r2_aitchison_class
    }
