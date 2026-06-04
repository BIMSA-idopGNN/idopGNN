import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
 
def load_graph_data(arg_name=None, data_dir="./data/"):
    """
    Load graph data and prepare PyTorch Geometric Data object with train/val/test masks.

    Args:
        data_dir (str): Directory containing the CSV files. Defaults to "./data/".

    Returns:
        data (torch_geometric.data.Data): Graph data with x, edge_index, edge_attr, y, and masks.
        train_label_df (pd.DataFrame): Selected training labels with sample names.
        val_label_df (pd.DataFrame): Validation labels with sample names.
        val_x (pd.DataFrame): Validation features with sample names.
        test_label_df (pd.DataFrame): Test labels with sample names.
        test_x (pd.DataFrame): Test features with sample names.
    """

    # File paths
    EDGE_INDEX_FILE = f"{data_dir}{arg_name}_edgeindex_data.csv"
    EDGE_ATTR_FILE = f"{data_dir}process data/{arg_name}/{arg_name}_edgefeature_data.csv"
    NODE_FEATURE_FILE = f"{data_dir}{arg_name}_nodesfeature_data.csv"
    LABEL_FILE = f"{data_dir}{arg_name}_func_nodeslabel_data.csv"

    # Load edge index
    edge_index_df = pd.read_csv(EDGE_INDEX_FILE)
    edge_index_np = np.array([
        edge_index_df['From_numeric'].values,
        edge_index_df['To_numeric'].values
    ])
    edge_index = torch.tensor(edge_index_np, dtype=torch.long) - 1

    # Load edge attributes
    edge_attr_df = pd.read_csv(EDGE_ATTR_FILE)
    edge_attr = torch.tensor(edge_attr_df['x'].values, dtype=torch.float).view(-1, 1).abs()

    # Load node features
    x_df = pd.read_csv(NODE_FEATURE_FILE)
    sample_names = x_df.iloc[:, 0]
    x = torch.tensor(x_df.iloc[:, 1:].values, dtype=torch.float)

    # Load labels
    y_df = pd.read_csv(LABEL_FILE)
    y = pd.DataFrame(y_df.iloc[:, 1:].values)

    # Split indices
    non_zero_counts = y.apply(lambda row: (row != 0).sum(), axis=1)
    non_zero_idx = non_zero_counts[non_zero_counts != 0].index

    if arg_name == "XD":
        seed = 42
    else:
        seed =1

    train_idx, temp_idx = train_test_split(non_zero_idx, test_size=0.4, random_state=seed)
    test_idx, val_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)

    # Filter columns with ones counts
    ones_count = y.apply(lambda col: (col == 1).sum())
    cols_to_drop_small = ones_count[ones_count.isin([1, 2, 3])].index

    max_ones = ones_count.max()
    cols_to_drop_max = ones_count[ones_count == max_ones].index

    cols_to_drop = cols_to_drop_small.union(cols_to_drop_max)

    y = y.drop(columns=cols_to_drop)

    # Save cleaned labels
    y_label = pd.concat([sample_names, y], axis=1)

    # Set seed
    np.random.seed(1)
    # Function to select rows per column
    def select_rows_for_column(column: pd.Series, train_set: pd.Index) -> np.ndarray:
        ones_idx = column[column == 1].index.intersection(train_set)
        zeros_idx = column[column == 0].index.intersection(train_set)
        selected_ones = np.random.choice(ones_idx, 2, replace=True)
        selected_zeros = np.random.choice(zeros_idx, 2, replace=True)
        return np.concatenate([selected_ones, selected_zeros])

    # Select rows for training
    selected_rows_all = []
    y_train = y.iloc[train_idx]
    for col in y_train.columns:
        selected_rows = select_rows_for_column(y[col], train_idx)
        selected_rows_all.extend(selected_rows)
    unique_selected_rows = np.unique(selected_rows_all)

    # Convert x to DataFrame for easy indexing
    x_df_tensor = pd.DataFrame(x.numpy(), index=sample_names)

    # Prepare label DataFrames
    train_label = y.iloc[unique_selected_rows]
    train_label_df = pd.concat([sample_names[unique_selected_rows], train_label], axis=1)

    val_label = y.iloc[val_idx]
    val_label_df = pd.concat([sample_names[val_idx], val_label], axis=1)
    val_x = pd.concat([sample_names[val_idx], x_df_tensor.iloc[val_idx]], axis=1)

    test_idx = np.sort(test_idx)
    test_label = y.iloc[test_idx]
    test_label_df = pd.concat([sample_names[test_idx], test_label], axis=1)
    test_x = pd.concat([sample_names[test_idx], x_df_tensor.iloc[test_idx]], axis=1)

    # Print info
    print(f"Test set indices: {test_idx}")
    print(f"Validation set indices: {val_idx}")
    print(f"Selected training rows: {unique_selected_rows}")

    # Construct masks
    train_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    train_mask[unique_selected_rows] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # Convert labels to tensor
    y_tensor = torch.tensor(y.values, dtype=torch.float)

    # Create PyG data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return data

