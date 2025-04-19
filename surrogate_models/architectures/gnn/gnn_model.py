# Copyright 2025 Fraunhofer Institute for Open Communication Systems FOKUS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import torch.nn.functional as F

from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from tqdm import tqdm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention


class GNN(torch.nn.Module):
    """
    A multi-layer Graph Neural Network (GNN) module for generating node representations.

    Args:
        num_layer (int): Number of GNN layers.
        emb_dim (int): Dimensionality of node embeddings.
        edge_attr_dim (int): Dimensionality of edge attributes.
        num_node_features (int): Number of features for each node.
        JK (str): Jump knowledge method ('last' or 'sum') to aggregate node representations.
        drop_ratio (float): Dropout rate used during training.
        gnn_type (str): Type of GNN to use; currently supports "gin" (other types could be added).

    Returns:
        Tensor: Final node representations.
    """
    def __init__(self, num_layer, emb_dim, edge_attr_dim, num_node_features, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.gnn_type = gnn_type

        # Initial linear encoding of node features into embedding space.
        self.node_encoder = torch.nn.Linear(num_node_features, emb_dim)

        # Construct a list of GNN layers.
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GCNConv(emb_dim=emb_dim, edge_attr_dim=edge_attr_dim, aggr="add"))

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass through the GNN layers.

        Args:
            x (Tensor): Node features of shape [num_nodes, num_node_features].
            edge_index (LongTensor): Graph connectivity in COO format.
            edge_attr (Tensor): Edge feature matrix.

        Returns:
            Tensor: Aggregated node embeddings.
        """
        # Encode node features.
        h = self.node_encoder(x)
        h_list = [h]
        # Pass through each GNN layer.
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            # Apply non-linearity for all layers except the last.
            if layer != self.num_layer - 1:
                h = F.relu(h)
            # Dropout for regularization.
            h = F.dropout(h, self.drop_ratio, training=self.training)
            h_list.append(h)

        # Aggregate using Jump Knowledge (JK) scheme.
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = torch.sum(torch.stack(h_list[1:]), dim=0)
        else:
            raise ValueError("Invalid JK mode.")
        return node_representation


class GCNConv(MessagePassing):
    """
    A customized Graph Convolution (GCN) layer with edge attribute encoding and self-loop handling.

    Args:
        emb_dim (int): Dimensionality of node embeddings.
        edge_attr_dim (int): Dimensionality of edge attributes.
        aggr (str): Aggregation method (default: "add").
        input_layer (bool): If True, use an embedding layer for input node features.
    """
    def __init__(self, emb_dim, edge_attr_dim, aggr="add", input_layer=False):
        super(GCNConv, self).__init__()
        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_encoder = torch.nn.Linear(edge_attr_dim, emb_dim)
        self.input_layer = input_layer
        if self.input_layer:
            # Initialize an embedding for input nodes (if necessary).
            self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
            torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)
        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        """
        Compute normalization coefficients for graph convolution.

        Args:
            edge_index (LongTensor): Graph connectivity.
            num_nodes (int): Number of nodes.
            dtype: Data type for the tensor.

        Returns:
            Tensor: Normalization coefficients for each edge.
        """
        # Set edge weights to 1.
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        row, col = edge_index

        # Compute degree of each node.
        deg = torch.zeros((num_nodes,), dtype=dtype, device=edge_index.device)
        deg.index_add_(0, row, edge_weight)

        # Compute inverse square root of degrees.
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # Return normalized edge weights.
        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass through the GCN convolution layer.

        Args:
            x (Tensor): Input node features.
            edge_index (LongTensor): Graph connectivity.
            edge_attr (Tensor): Edge feature matrix.

        Returns:
            Tensor: Updated node features after message passing.
        """
        # Add self-loops to the graph.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Create attributes for self-loop edges: a vector of zeros with first element = 1.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = 1
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # Concatenate self-loop edge attributes.
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # Encode edge attributes.
        edge_embeddings = self.edge_encoder(edge_attr)

        # If using an input layer, convert node indices to embeddings.
        if self.input_layer:
            x = self.input_node_embeddings(x.to(torch.int64).view(-1, ))

        # Compute normalization factors.
        norm = self.norm(edge_index, x.size(0), x.dtype)

        # Transform node features.
        x = self.linear(x)
        # Propagate messages along edges.
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        """
        Message function for aggregating neighboring node features.

        Args:
            x_j (Tensor): Features from neighboring nodes.
            edge_attr (Tensor): Encoded edge attributes.
            norm (Tensor): Normalization coefficients.

        Returns:
            Tensor: Message to be aggregated.
        """
        # Multiply normalization factor with the sum of neighboring feature and edge attribute.
        return norm.view(-1, 1) * (x_j + edge_attr)


class RegGNN(torch.nn.Module):
    """
    Regression GNN model for predicting the fitness label of quantum circuits.

    Args:
        num_layer (int): Number of GNN layers.
        emb_dim (int): Dimensionality of node embeddings.
        edge_attr_dim (int): Dimensionality of edge attributes.
        num_node_features (int): Number of node features.
        JK (str): Jump knowledge method ("last" or "sum") for aggregating node representations.
        drop_ratio (float): Dropout rate.
        graph_pooling (str): Graph pooling method ("sum", "mean", "max", or "attention").
        gnn_type (str): Type of GNN (e.g., "gin").
        freeze_gnn (bool): If True, freeze the GNN backbone during training.
    """
    def __init__(self, num_layer, emb_dim, edge_attr_dim, num_node_features, JK="last", drop_ratio=0.5, graph_pooling="mean", gnn_type="gin", freeze_gnn=False):
        super(RegGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.edge_attr_dim = edge_attr_dim
        self.num_node_features = num_node_features
        self.gnn_type = gnn_type
        self.JK = JK
        self.freeze_gnn = freeze_gnn

        # Initialize the GNN component.
        self.gnn = GNN(num_layer, emb_dim, edge_attr_dim, num_node_features, JK="last", drop_ratio=drop_ratio, gnn_type=gnn_type)

        # Define the graph pooling layer based on the given method.
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

        # Final linear layers for graph-level prediction.
        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, 1)
        self.graph_pred_linear_proxy = torch.nn.Linear(self.emb_dim + 1, 1)

        # Optionally freeze the GNN layers.
        if self.freeze_gnn:
            for param in self.gnn.parameters():
                param.requires_grad = False

    def from_pretrained(self, model_file):
        """
        Load pretrained weights into the GNN component from a file.

        Args:
            model_file (str): Path to the pretrained model file.
        """
        pretrained_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
        model_dict = self.gnn.state_dict()
        # Load pretrained weights (allowing missing keys with strict=False).
        self.gnn.load_state_dict(pretrained_dict, strict=False)

    def forward(self, data):
        """
        Forward pass for graph-level regression prediction.

        Args:
            data (Data): A torch_geometric data object containing:
                - x: Node features.
                - edge_index: Graph connectivity.
                - edge_attr: Edge features.
                - batch: Batch indices for pooling.
                - proxy (optional): Extra features for proxy prediction.

        Returns:
            Tensor: Predictions of shape (batch_size,).
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        proxy = getattr(data, "proxy", None)

        # Obtain node representations via the GNN.
        node_representation = self.gnn(x, edge_index, edge_attr)
        # Pool node representations into a graph-level representation.
        graph_rep = self.pool(node_representation, batch)

        # If extra proxy information is present, concatenate and use a different linear layer.
        if proxy is not None:
            proxy = proxy.view(-1, 1)
            graph_rep = torch.cat([graph_rep, proxy], dim=1)
            out = self.graph_pred_linear_proxy(graph_rep)
        else:
            out = self.graph_pred_linear(graph_rep)
        return out.view(-1)  # Reshape to (batch_size,)

    def predict(self, data):
        """
        Generate clamped predictions for the input batch.

        Args:
            data (Data): A torch_geometric data object.

        Returns:
            Tensor: Predictions clamped between 0 and 1.
        """
        x = self.forward(data)
        # Clamp predictions to be within [0, 1].
        out = torch.clamp(x, 0, 1)
        return out

    @staticmethod
    def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
        """
        Train the model for one epoch.

        Args:
            model (RegGNN): The regression GNN model.
            dataloader (DataLoader): DataLoader for training data.
            optimizer (Optimizer): Optimizer for updating model parameters.
            loss_fn (function): Loss function.
            device (torch.device): Device to use for training.

        Returns:
            tuple: (avg_loss, spearman, r2) computed over the epoch.
        """
        model.train()
        total_loss, total_samples = 0.0, 0
        all_outputs, all_labels = [], []
        for batch in tqdm(dataloader, desc="Training"):
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            labels = batch.y
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
            all_outputs.append(output.detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_outputs = np.concatenate(all_outputs)
        all_labels = np.concatenate(all_labels)
        r2 = r2_score(all_labels, all_outputs)
        avg_loss = total_loss / total_samples
        spearman, _ = spearmanr(all_labels, all_outputs)

        return avg_loss, spearman, r2

    @staticmethod
    def evaluate(model, dataloader, loss_fn, device, desc="Evaluating"):
        """
        Evaluate the model on a dataset.

        Args:
            model (RegGNN): The regression GNN model.
            dataloader (DataLoader): DataLoader for evaluation data.
            loss_fn (function): Loss function.
            device (torch.device): Device to perform evaluation on.
            desc (str): Description for progress bar.

        Returns:
            tuple: (avg_loss, spearman, r2) computed over the evaluation dataset.
        """
        model.eval()
        total_loss, total_samples = 0.0, 0
        all_outputs, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                batch = batch.to(device)
                output = model.predict(batch)
                labels = batch.y
                loss = loss_fn(output, labels)
                total_loss += loss.item() * batch.num_graphs
                total_samples += batch.num_graphs
                all_outputs.append(output.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_outputs = np.concatenate(all_outputs)
        all_labels = np.concatenate(all_labels)
        r2 = r2_score(all_labels, all_outputs)
        avg_loss = total_loss / total_samples
        spearman, _ = spearmanr(all_labels, all_outputs)

        return avg_loss, spearman, r2
