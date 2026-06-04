import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj

from utils.phenotypic_predict.train_eva_utils import create_graph_data,split_indices

# Set device to CPU for computation 
device = torch.device('cpu')

# Define file paths for data loading
DATA_DIR = "./data/"
EDGE_INDEX_FILE = f"{DATA_DIR}all_data_edgeindex_data.csv"
EDGE_ATTR_FILE = f"{DATA_DIR}all_data_edgefeature_data.csv"
NODE_FEATURE_FILE = f"{DATA_DIR}all_data_nodesfeature_data.csv"
LABEL_FILE = f"{DATA_DIR}all_data_func_nodeslabel_data.csv"
EMBEDDING_FILE = f"{DATA_DIR}all_data_embeddings_tensor.csv"

# Load phenotypic factor labels
factor = pd.read_csv(LABEL_FILE, index_col=0)
factor.columns = factor.columns.str.replace('.', ' ')

# Define target variable names and extract corresponding data
names = ['Acidity', 'Sugar', 'Moisture', 'Starch', 'Acetic Acid', 'Ethanol', 'Lactic Acid']
Acidity = factor[names]

# Load and process edge index data for graph structure
edge_index_data = pd.read_csv(EDGE_INDEX_FILE)
edge_index_data = np.array([
    edge_index_data['From_numeric'].values,
    edge_index_data['To_numeric'].values
])

# Convert to tensor
edge_index = torch.tensor(edge_index_data, dtype=torch.long)
edge_index = edge_index - 1

# Load node feature data (species abundance)
rawabundance = pd.read_csv(NODE_FEATURE_FILE)
species_names = rawabundance.iloc[:, 0]
abundance_df = rawabundance.iloc[:, 1:]
abundance_df.index = species_names
abundance = rawabundance.iloc[:, 1:].values
sample_names = rawabundance.columns[1:]
abundance_df = abundance_df.T

# Load and process edge attributes
edge_attr = pd.read_csv(EDGE_ATTR_FILE)
edge_attr = torch.tensor(edge_attr['x'].values, dtype=torch.float32).abs()

# Load node embeddings
embeding = pd.read_csv(EMBEDDING_FILE)
embeding_rawname = embeding.iloc[:, 0]
embeding = embeding.iloc[:, 1:]
embeding.index = embeding_rawname
embeding = embeding.loc[species_names] # Align embeddings with species names
in_channels = embeding.shape[1]

# Process target variables (phenotypic measurements)
y = Acidity.loc[sample_names]
y = y / 100
y = pd.DataFrame(y)

res = 1 - y.sum(axis=1)
y = pd.concat([y, res], axis=1)
out_channels = y.shape[1]


def smithson_verkuilen_transform(y):
    """
    Apply Smithson-Verkuilen transformation to compositional data.
    Transforms bounded [0,1] data to unbounded range for better modeling.
    
    Args:
        y (pd.DataFrame): Compositional data with values in [0,1]
    
    Returns:
        pd.DataFrame: Transformed data
    """
    N, C = y.shape
    y_transformed = (y * (N - 1) + 1 / C) / N
    return y_transformed

# Apply transformation to target variables
y = smithson_verkuilen_transform(y)


def graph_data(seed=None, batch_size_train=64, batch_size_val=34, batch_size_test=40, embedding=embeding,edge_attr=edge_attr, create_graph_data=create_graph_data, device=device):
    """
    Split abundance and label data into train/val/test sets, standardize features,
    convert to PyTorch tensors, create graph datasets, and return DataLoaders.

    Args:
        abundance_df (pd.DataFrame): Feature matrix (samples x features)
        y (pd.DataFrame): Label matrix
        edge_index (torch.Tensor): Graph edge indices
        embedding: Optional embedding information for graph creation
        edge_attr: Edge attributes
        create_graph_data: Function to convert sample to a graph object
        device: torch device
        seed (int): Random seed for reproducibility
        batch_size_train (int): Batch size for training DataLoader
        batch_size_val (int): Batch size for validation DataLoader
        batch_size_test (int): Batch size for test DataLoader

    Returns:
        data
    """

    # Split indices
    train_idx, val_idx, test_idx = split_indices(abundance_df)

    # Split data
    X_train, X_val, X_test = abundance_df.loc[train_idx], abundance_df.loc[val_idx], abundance_df.loc[test_idx]
    Y_train, Y_val, Y_test = y.loc[train_idx], y.loc[val_idx], y.loc[test_idx]

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    Y_train = torch.tensor(np.stack(Y_train.values), dtype=torch.float32).to(device)
    Y_val = torch.tensor(np.stack(Y_val.values), dtype=torch.float32).to(device)
    Y_test = torch.tensor(np.stack(Y_test.values), dtype=torch.float32).to(device)

    # Create graph datasets
    graphs_train = [create_graph_data(sample, edge_index.to(device), l, embedding, edge_attr, device)
                    for sample, l in zip(X_train, Y_train)]
    graphs_val = [create_graph_data(sample, edge_index.to(device), l, embedding, edge_attr, device)
                  for sample, l in zip(X_val, Y_val)]
    graphs_test = [create_graph_data(sample, edge_index.to(device), l, embedding, edge_attr, device)
                   for sample, l in zip(X_test, Y_test)]

    # Move graphs to device
    graphs_train = [graph.to(device) for graph in graphs_train]
    graphs_val = [graph.to(device) for graph in graphs_val]
    graphs_test = [graph.to(device) for graph in graphs_test]

    # Create DataLoaders
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(graphs_train, batch_size=batch_size_train, shuffle=True, generator=g)
    val_loader = DataLoader(graphs_val, batch_size=batch_size_val, shuffle=False)
    test_loader = DataLoader(graphs_test, batch_size=batch_size_test, shuffle=False)

    A0 = to_dense_adj(edge_index, edge_attr=edge_attr)[0].squeeze(-1)
    return {"train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "embeding": embeding,
            "A0": A0}
