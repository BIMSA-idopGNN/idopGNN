import sys
import json
import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from utils.gnnconv import gnnConv
from torch_geometric.utils import dense_to_sparse, to_dense_batch

# Parse command line argument for dataset / task type
if len(sys.argv) >= 2:
    arg_name = sys.argv[1]
else:
    arg_name = "XD"

print("User input:", arg_name)

class GraphLearner(nn.Module):
    """
    Learnable graph structure module.
    Combines MLP-based similarity with distance-based kernel weighting,
    and optionally merges with a prior adjacency matrix.
    """

    def __init__(self, in_channels, hidden_channels, kappa=None, ealpha=None, topk=None):
        super().__init__()

        # Hyperparameters for graph learning
        self.kappa = kappa # Bandwidth parameter for Gaussian kernel
        self.alpha = ealpha # Weight for combining learned and prior adjacency
        self.topk = topk # Top-k sparsification

        # MLP for learning node similarity scores
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        # Learnable projection matrix for distance computation
        self.W = nn.Parameter(torch.randn(in_channels, in_channels))

    def forward(self, X, A0=None, batch=None):

        """
        Args:
            X: Node feature matrix [N, F]
            A0: Optional prior adjacency matrix [N, N]
            batch: Batch vector for graph-level tasks
        Returns:
            A_learned: Learned adjacency matrix [N, N]
        """
        if batch is None:
            # Node-level task: process all nodes together
            N = X.size(0)
            x_left = X.unsqueeze(1).expand(N, N, -1)  # [N, N, F]
            x_right = X.unsqueeze(0).expand(N, N, -1)  # [N, N, F]
            diff = X.unsqueeze(1) - X.unsqueeze(0)  # [N, N, F]

        else:
            # Graph-level task: process batched graphs

            x_dense, mask = to_dense_batch(X, batch)
            B, N_max, F = x_dense.size()

            # Compute mean features for each graph position          
            valid_mask = mask.unsqueeze(-1)
            x_valid = x_dense * valid_mask
            valid_count = torch.clamp(valid_mask.sum(dim=0), min=1)  # [N_max, 1]
            x_mean = x_valid.sum(dim=0) / valid_count

            N = mask.any(dim=0).sum().item()
            X_mean = x_mean[:N]  # [n, F]

            x_left = X_mean.unsqueeze(1).expand(N, N, F)
            x_right = X_mean.unsqueeze(0).expand(N, N, F)

            diff = X_mean.unsqueeze(1) - X_mean.unsqueeze(0)

        # Concatenate node pairs for MLP processing      
        pair_features = torch.cat([x_left, x_right], dim=-1)  # [N, N, 2F]

        # Compute similarity scores using MLP      
        mlp_scores = self.mlp(pair_features).squeeze(-1)  # [N, N]

        # Compute distance-based scores using Mahalanobis distance    
        M = self.W @ self.W.T  # [F, F]
        diff_proj = torch.matmul(diff, M)  # [N, N, F]
        dist = (diff_proj * diff).sum(dim=-1)  # [N, N]
        dist_scores = torch.exp(-dist / (2 * self.kappa ** 2))

        # Combine MLP and distance scores      
        A_learned = torch.sigmoid(mlp_scores) * dist_scores

        # Apply top-k sparsification if specified      
        if self.topk is not None:
            k = min(self.topk, N)
            topk_values, topk_indices = torch.topk(A_learned, k, dim=1)
            mask_t = torch.zeros_like(A_learned)
            row_indices = torch.arange(N, device=X.device).unsqueeze(1).expand(-1, k)
            mask_t[row_indices, topk_indices] = 1.0
            A_learned = A_learned * mask_t

        # Combine with prior adjacency
        if A0 is not None:
            A_learned = self.alpha * A_learned + (1 - self.alpha) * A0

        if batch is not None:
            A_list = []
            for b in range(B):
                n_b = mask[b].sum().item()
                A_b = A_learned[:n_b, :n_b].clone()

                pad = torch.zeros(N_max, N_max, device=X.device)
                pad[:n_b, :n_b] = A_b
                A_list.append(pad)

            A_batch = torch.stack(A_list, dim=0)  # [B, N_max, N_max]

            return A_batch, mask
        else:
            return A_learned


class idopGNN(nn.Module):
    """
    Graph Convolutional Network with learnable structure.
    Each layer learns an adaptive adjacency via GraphLearner,
    followed by a graph convolution operation.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=None, ealpha=None, nalpha=None,
                 kappa=None, topk=None, dropout=None,arg_name=None):
        super().__init__()
        assert num_layers >= 2, "warning"

        # Initialize GNN layers and graph learners                   
        self.convs = nn.ModuleList()
        self.graph_learners = nn.ModuleList()

        # Store hyperparameters                   
        self.num_layers = num_layers
        self.nalpha = nalpha # Parameter for GNN convolution
        self.ealpha = ealpha # Parameter for graph learner
        self.kappa = kappa # Bandwidth for Gaussian kernel
        self.topk = topk # Top-k sparsification
        self.dropout = dropout

        # First layer                   
        self.convs.append(gnnConv(in_channels, hidden_channels, nalpha=self.nalpha))
        self.graph_learners.append(GraphLearner(
            in_channels, hidden_channels,
            kappa=self.kappa, ealpha=self.ealpha, topk=self.topk
        ))

        # Intermediate layers                   
        for _ in range(num_layers - 2):
            self.convs.append(gnnConv(hidden_channels, hidden_channels, nalpha=self.nalpha))
            self.graph_learners.append(GraphLearner(
                hidden_channels, hidden_channels,
                kappa=self.kappa, ealpha=self.ealpha, topk=self.topk
            ))

        # Final layer                   
        self.convs.append(gnnConv(hidden_channels, out_channels, nalpha=self.nalpha))
        self.graph_learners.append(GraphLearner(
            hidden_channels, hidden_channels,
            kappa=self.kappa, ealpha=self.ealpha, topk=self.topk
        ))

        # Final linear layer for graph-level tasks (phenotypic prediction)                  
        if arg_name not in ["XD", "ZD", "YD", "ED"]:
            self.final = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, A0=None, batch=None):

        """
        Forward pass through GNN layers with adaptive structure learning.
        Args:
            x: Node feature matrix [N, F]
            A0: Optional prior adjacency matrix [N, N]
            batch: Batch vector for graph-level tasks
        Returns:
            x: Node embeddings or output logits [N, out_channels]
        """
        if batch is None:
            # Node-level task (function prediction)         
            for i, conv in enumerate(self.convs):
                # Learn adaptive adjacency matrix              
                A = self.graph_learners[i](x, A0=A0)
                edge_index, edge_weight = dense_to_sparse(A)

                # Apply graph convolution              
                x = conv(x, edge_index, edge_weight=edge_weight)

                # Apply non-linearity and dropout for intermediate layers              
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                  
            # Final activation for node-level task
            x = torch.sigmoid(x)

        else:
            # Graph-level task (phenotypic prediction)          
            for i in range(self.num_layers-1):
                # Learn adaptive adjacency for batched graphs          
                A_batch, mask = self.graph_learners[i](x, A0=A0, batch=batch)
              
                edge_index_list, edge_weight_list = [], []
                cum_nodes = 0
                B, N_max, _ = A_batch.size()
                for b in range(B):
                    n = mask[b].sum().item()
                    A = A_batch[b, :n, :n]
                    edge_index_b, edge_weight_b = dense_to_sparse(A)
                    edge_index_b = edge_index_b + cum_nodes
                    cum_nodes += n
                    edge_index_list.append(edge_index_b)
                    edge_weight_list.append(edge_weight_b)

                edge_index = torch.cat(edge_index_list, dim=1)
                edge_weight = torch.cat(edge_weight_list)

                # Apply graph convolution              
                x = self.convs[i](x, edge_index, edge_weight=edge_weight)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)

            # Global pooling and final projection for graph-level task          
            x = global_mean_pool(x, batch)
            x = self.final(x)
            x = torch.exp(x)

        return x
def set_random_seed(seed):
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load configuration from JSON file
with open("./utils/config.json", "r") as f:
    configs = json.load(f)


def training_evaluation(set_random_seed, configs, arg_name):
    """
    Model training and evaluation pipeline.
    Handles both function prediction and phenotypic prediction tasks.
    """

    if len(arg_name) == 2:
        # Function prediction task
        print("Function prediction task selected.")
        from preprocess.function_predict.data_loader import load_graph_data
        from utils.function_predict.train_eva_utils import train_class_early_stopping_loss, evaluate_metrics_class

        # Load graph data for function prediction
        data = load_graph_data(arg_name)
        in_channels = data.x.shape[1]
        out_channels = data.y.size(1)
        criterion = nn.BCELoss()

        if arg_name == "XD":
            config = configs["XD"]
        elif arg_name == "ZD":
            config = configs["ZD"]
        elif arg_name == "YD":
            config = configs["YD"]
        elif arg_name == "ED":
            config = configs["ED"]

    else:
        # Phenotypic prediction task (regression)      
        print("Phenotypic prediction task selected.")
        from preprocess.phenotypic_predict.data_loader import graph_data,in_channels,out_channels
        from utils.phenotypic_predict.train_eva_utils import train_regression_early_stopping_loss, neg_log_likelihood, evaluate_metrics_regression
        criterion = neg_log_likelihood

        config = configs["all"]

    # Initialize lists to store evaluation metrics  
    auroc_list, auprc_list, micro_f1_list, macro_f1_list = [], [], [], []
    r2_list, aitchison_list, r2_aitchison_list, r2_class_list, aitchison_class_list, r2_aitchison_class_list = [], [], [], [], [], []

    # Run training and evaluation for 5 random seeds  
    for seed in range(1, 6):

        set_random_seed(seed)
        print(f"Training with random seed {seed}")

        model = idopGNN(
            in_channels=in_channels,
            hidden_channels=config['hidden_channels'],
            out_channels=out_channels,
            num_layers=config['num_layers'],
            ealpha=config['ealpha'],
            nalpha=config['nalpha'],
            kappa=config['kappa'],
            topk=config['topk'],
            dropout=config['dropout'],
            arg_name = arg_name
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        if len(arg_name)==2:
            # Function prediction training and evaluation
            _, model = train_class_early_stopping_loss(
                model, data, optimizer, criterion, max_epochs=config['max_epochs'], patience=config['patience']
            )

            metrics = evaluate_metrics_class(model, data.x, data.edge_index, data.edge_attr, data.y, data.test_mask)

            auroc_list.append(metrics['auroc'])
            auprc_list.append(metrics['auprc'])
            micro_f1_list.append(metrics['micro_f1'])
            macro_f1_list.append(metrics['macro_f1'])
        else:
            # Phenotypic prediction training and evaluation
            data = graph_data(seed)

            _, model = train_regression_early_stopping_loss(
                model, data, optimizer, criterion, max_epochs=config['max_epochs'], patience=config['patience']
            )

            metrics = evaluate_metrics_regression(model, data)

            r2_list.append(metrics['r2'])
            aitchison_list.append(metrics['aitchison'])
            r2_aitchison_list.append(metrics['r2_aitchison'])
            r2_class_list.append(metrics['r2_class'])
            aitchison_class_list.append(metrics['aitchison_class'])
            r2_aitchison_class_list.append(metrics['r2_aitchison_class'])

    # Process and display results based on task type  
    if len(arg_name) !=2:

        results = {
            'r2': (np.mean(r2_list), np.std(r2_list)),
            'aitchison': (np.mean(aitchison_list), np.std(aitchison_list)),
            'r2_aitchison': (np.mean(r2_aitchison_list), np.std(r2_aitchison_list)),
            'r2_class': r2_class_list,
            'aitchison_class': aitchison_class_list,
            'r2_aitchison_class': r2_aitchison_class_list,
        }

        print("Final Results:")
        print("R²: {:.4f}±{:.4f}".format(results['r2'][0], results['r2'][1]))
        print("Aitchison: {:.4f}±{:.4f}".format(results['aitchison'][0], results['aitchison'][1]))
        print("R² Aitchison: {:.4f}±{:.4f}".format(results['r2_aitchison'][0], results['r2_aitchison'][1]))

        # Calculate per-class statistics      
        R_squared_stack = np.array(results['r2_class'])
        aitchison_stack = np.array(results['aitchison_class'])
        r2_aitchison_stack = np.array(results['r2_aitchison_class'])

        R_squared_mean = np.mean(R_squared_stack, axis=0)
        R_squared_std = np.std(R_squared_stack, axis=0)

        aitchison_mean = np.mean(aitchison_stack, axis=0)
        aitchison_std = np.std(aitchison_stack, axis=0)

        r2_aitchison_mean = np.mean(r2_aitchison_stack, axis=0)
        r2_aitchison_std = np.std(r2_aitchison_stack, axis=0)

        for mean, std in zip(R_squared_mean, R_squared_std):
            print(f"R²: {mean:.4f}±{std:.4f}")

        for mean, std in zip(aitchison_mean, aitchison_std):
            print(f"Aitchison: {mean:.4f}±{std:.4f}")

        for mean, std in zip(r2_aitchison_mean, r2_aitchison_std):
            print(f"R² Aitchison: {mean:.4f}±{std:.4f}")

    else:
        # Function prediction results
        results = {
            'auroc': (np.mean(auroc_list), np.std(auroc_list)),
            'auprc': (np.mean(auprc_list), np.std(auprc_list)),
            'micro_f1': (np.mean(micro_f1_list), np.std(micro_f1_list)),
            'macro_f1': (np.mean(macro_f1_list), np.std(macro_f1_list)),
        }
        print("Final Results:")
        print("AUROC: {:.2%}±{:.2%}".format(results['auroc'][0], results['auroc'][1]))
        print("AUPRC: {:.2%}±{:.2%}".format(results['auprc'][0], results['auprc'][1]))
        print("Micro F1: {:.2%}±{:.2%}".format(results['micro_f1'][0], results['micro_f1'][1]))
        print("Macro F1: {:.2%}±{:.2%}".format(results['macro_f1'][0], results['macro_f1'][1]))

    return results


if __name__ == "__main__":
    # Execute training and evaluation pipeline
    results = training_evaluation(set_random_seed=set_random_seed, configs=configs, arg_name=arg_name)


