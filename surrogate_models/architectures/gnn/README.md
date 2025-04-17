# Regression GNN Model

This repository contains a PyTorch-based implementation of a Graph Neural Network (GNN) model for regression tasks. The model is designed to predict circuit fitness and includes custom GNN layers, training, and evaluation utilities.

## Overview

- **GNN**: Builds node embeddings using multiple GCNConv layers and supports jump knowledge (JK) aggregation ("last" or "sum").
- **GCNConv**: A custom graph convolution layer that adds self-loops, encodes edge attributes, and applies normalization.
- **RegGNN**: The full regression model, incorporating a GNN backbone, graph pooling (sum, mean, max, or attention), and a final prediction layer.
- **Training & Evaluation**: Utility methods `train_one_epoch` and `evaluate` are provided to train the model and assess its performance using metrics like R², loss, and Spearman correlation.

## Usage

1. **Model Initialization**:  
   Create a `RegGNN` instance with your desired configuration parameters:
   ```python
   model = RegGNN(
       num_layer=5,
       emb_dim=128,
       edge_attr_dim=...,
       num_node_features=...,
       JK="last",
       drop_ratio=0.5,
       graph_pooling="mean"
   )
   

2. **Forward Pass & Prediction:**
    Pass a batch of data (with `x`, `edge_index`, `edge_attr`, and `batch`) to obtain predictions:
    ```ptyhon
    output = model(data)
    predicted = model.predict(data)

3. **Evaluation:**
Use `RegGNN.evaluate(model, dataloader, loss_fn, device)` to compute `MSE`, `Spearman correlation`, and `R2` score on a validation or test dataset.