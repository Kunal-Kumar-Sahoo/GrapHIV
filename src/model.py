import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

torch.manual_seed(42)


class MoleculeNet(nn.Module):
    def __init__(self, feature_size, model_params):
        """
        feature_size (int): Dimension of the input vector
        model_params (str): Path to the configuratipn file defining parameters of the model
        """
        super().__init__()
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        self.n_layers = model_params["model_layers"]
        dropout_rate = model_params["model_dropout_rate"]
        top_k_ratio = model_params["model_top_k_ratio"]
        self.top_k_every_n = model_params["model_top_k_every_n"]
        dense_neurons = model_params["model_dense_neurons"]
        edge_dim = model_params["model_edge_dim"]

        self.conv_layers = nn.ModuleList([])
        self.transf_layers = nn.ModuleList([])
        self.pooling_layers = nn.ModuleList([])
        self.bn_layers = nn.ModuleList([])

        # Transformation layers
        self.conv1 = gnn.TransformerConv(feature_size,
                                         embedding_size,
                                         heads=n_heads,
                                         dropout=dropout_rate,
                                         edge_dim=edge_dim,
                                         beta=True)
        
        self.transf1 = nn.Linear(embedding_size * n_heads, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)

        # Other layers
        for i in range(self.n_layers):
            self.conv_layers.append(
                gnn.TransformerConv(embedding_size,
                                    embedding_size,
                                    heads=n_heads,
                                    dropout=dropout_rate,
                                    edge_dim=edge_dim,
                                    beta=True))
            self.transf_layers.append(nn.Linear(embedding_size * n_heads, embedding_size))
            self.bn_layers.append(nn.BatchNorm1d(embedding_size))

            if i % self.top_k_every_n == 0:
                self.pooling_layers.append(gnn.TopKPooling(embedding_size, ratio=top_k_ratio))

        # Linear layers
        self.linear1 = nn.Linear(embedding_size * 2, dense_neurons)
        self.linear2 = nn.Linear(dense_neurons, dense_neurons // 2)
        self.linear3 = nn.Linear(dense_neurons // 2, 1)

    def forward(self, x, edge_attr, edge_index, batch_index):
        # Initial transformation
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        # Holds the intermediate graph representations
        global_representations = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)

            # Aggregate the last layer
            if i % self.top_k_every_n == 0 or i == self.n_layers:
                x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[i // self.top_k_every_n](
                    x, edge_index, edge_attr, batch_index)
                # Add current representations
                global_representations.append(torch.cat([
                    gnn.global_max_pool(x, batch_index),
                    gnn.global_mean_pool(x, batch_index)
                ], dim=1))
        
        x = sum(global_representations)

        # Output block
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)

        return x