# **idopGNN**

<img width="1299" height="742" alt="image" src="https://github.com/user-attachments/assets/724ab468-1b94-4fea-ba0a-74e477467982" />


## **Introduction**
 
**idopGNN** is an integrated framework for **microbial functional prediction** and **phenotypic prediction**.
It begins with the construction of an **ecologically informed prior network (idopNetwork)**, reconstructed using evolutionary game theory and **quasi-dynamic ODEs** derived from allometric scaling laws.
In this network, nodes represent microbial taxa, and edges encode full information ecological interactions.

A **graph structure learning module** then adaptively refines this prior network by incorporating data-driven relational cues to obtain a task-specific optimized topology.
On top of the refined graph, a **dual-residual GNN** performs representation learning

* **Node-level residuals** alleviate over-smoothing,
* **Edge-level residuals** preserve biologically meaningful interactions.

This design ensures **predictions**, even under limited labeled samples.

---

## **Environment**

| Software     | Version |
| ------------ | ------- |
| R            | 4.4.2   |
| Python       | 3.12.3  |
| PyTorch      | 2.7.1   |
| scikit-learn | 1.6.1   |
| numpy        | 2.1.3   |
| pandas       | 2.2.3   |
| Ubuntu       | 22.04.4 |

---

## **Repository Structure Overview**

The repository implements the full **idopGNN pipeline**, consisting of data preprocessing, network reconstruction, graph learning, and task-specific prediction.

```
idopGNN/
├── data/
├── preprocess/
├── idopNetwork/
├── idopGNN/
└── utils/
```

---

## **`data/`**

Stores all **raw** and **processed** datasets used throughout the pipeline.

---

## **`preprocess/`**

Contains scripts for data cleaning, functional annotation, and preparing inputs for GNN models.

### **Included Codes**

#### **1. `raw_data_clean.R`**

Cleans, filters, and normalizes microbial abundance data.

```bash
Rscript preprocess/raw_data_clean.R \
  --input raw_data \
  --output ./data/XD.csv \
  --output ./data/ZD.csv \
  --output ./data/YD.csv \
  --output ./data/ED.csv \
  --output ./data/all_data.csv \ # Phenotypic predict unified four-stage microbiological data
  --output ./data/df_otu.csv \
  --output ./data/df_info_new.csv
```

#### **2. `function_label.R`**

Generates functional annotations for the four stages (XD/ZD/YD/ED).

```bash
Rscript preprocess/Function_Label.R \
  --input ./data/raw_data_clean_result \
  --output ./data/XD_function.csv \
  --output ./data/ZD_function.csv \
  --output ./data/YD_function.csv \
  --output ./data/ED_function.csv
```

#### **3. Data loaders**

* `function_predict/data_loader.py` — Node-level classification
* `phenotypic_predict/data_loader.py` — Graph-level regression
* `phenotypic_predict/embedding.py` — Convert microbial fasta sequences into embeddings

---

## **`idopNetwork/`**

Implements **idopNetwork reconstruction**, including allometric scaling fits, qdODE solving, and extraction of graph-compatible data.

### **Main Components**

#### **1. `allometric scaling + zero-inflated modeling`**

```bash
Rscript IdopNetwork/allometric_scaling_law_fit.R "XD"\
  --input ./data/XD \
  --output ./data/XD_fit_result.RData
  --output ./data/XD_fit_result_plot.RData
```

Replace XD with ZD/YD/ED/all_data for other datasets.
Adjust the fitting method(BFGS OR Nelder-Mead) appropriately based on the fitting results.

#### **2. `qdMODE_model_solving.R`**

Solves quasi-dynamic ODEs to infer ecological interactions.

```bash
Rscript IdopNetwork/qdMODE_model_solving.R "XD"\
  --input ./data/XD_fit_result.RData \
  --output ./data/XD_qdODE_result.RData
```

#### **3. `get_net_data.R`**

Converts inferred networks into edge lists and feature matrices for GNNs.

```bash
Rscript IdopNetwork/get_net_data.R "XD"\
  --input ./data/XD_qdODE_result.RData \
  --output ./data/XD_nodesfeature_data.csv \
  --output ./data/XD_edgeindex_data.csv \
  --output ./data/XD_edgefeature_data.csv \
  --output ./data/XD_func_nodeslabel_data.csv \
```

---

## **`idopGNN/`**

Implements the **idopGNN** architectures for functional and phenotypic prediction.

### **Models**

#### **1. `idopGNN.py`**

```bash
python -m idopGNN.idopGNN "XD"\
  --input GraphData \
  --output PredictionResults
```

You may specify datasets such as `XD`, `ZD`, `YD`, `ED`, or `all_data` by appending them to the command arguments.

---

## **`utils/`**

Utility functions supporting graph learning and evaluation.

* **`gnnconv.py`** — Customized GNN convolution with residual pathways
* **`config.json`** — Stores all hyperparameters and model configurations used for training.
* **`function_predict/train_eva_utils.py`** — Loss + evaluation metrics for functional prediction
* **`phenotypic_predict/train_eva_utils.py`** — Loss + evaluation metrics for phenotypic prediction

---

## **Workflow Summary**

You may append **XD / ZD / YD / ED / all_data** to scripts to specify the dataset.

1. **Preprocessing**
   `raw_data_clean.R` → `function_label.R`
2. **Network Reconstruction**
   Allometric fitting → qdODE solving → network extraction
3. **Data Preparation**
   `function_predict/data_loader.py` or `phenotypic_predict/data_loader.py`
4. **Model Training & Evaluation**
   `idopGNN.py`

