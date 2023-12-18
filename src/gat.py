import torch
from torch import nn
import numpy as np

"""
The majority of this file was written by Aleksa Gordić and shared on Github in the
repository gordicaleksa/pytorch-GAT under the MIT license.
"""


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network (GAT) based on work by P. Veličković et al. in
    "Graph Attention Networks" (https://arxiv.org/abs/1710.10903). From the
    LFAV paper:

    "We propose a graph attention network (GAT) based model to refine event-aware
    snippet features, Fig. 6 shows its detailed architecture. For a layer in the
    model, all GAT modules in the same layer share the same weights but use
    category-aware graph structures to aggregate snippet features in different events.
    For each GAT module, we obtain refined event-aware snippet features. Then the
    output of the layer is the average output of all GAT modules."

    Args:
        feature_dim: dimension of the input features
        num_heads: number of attention heads
        num_classes: number of classes in the dataset
    """

    def __init__(
        self,
        num_of_layers,
        num_heads_per_layer,
        num_features_per_layer,
        dropout,
        add_skip_connection=True,
        bias=True,
        log_attention_weights=False,
    ):
        super().__init__()
        assert (
            num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1
        ), "Enter valid arch params."

        num_heads_per_layer = [1] + num_heads_per_layer

        gat_layers = [
            GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],
                num_out_features=num_features_per_layer[i + 1],
                num_of_heads=num_heads_per_layer[i + 1],
                concat=True if i < num_of_layers - 1 else False,
                activation=nn.ELU() if i < num_of_layers - 1 else None,
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights,
            )
            for i in range(num_of_layers)
        ]

        self.gat_net = nn.Sequential(*gat_layers)

    def forward(self, data):
        return self.gat_net(data)


class GATLayer(nn.Module):
    src_nodes_dim = 0
    trg_nodes_dim = 1

    nodes_dim = 0
    head_dim = 1

    def __init__(
        self,
        num_in_features,
        num_out_features,
        num_of_heads,
        concat=True,
        activation=nn.ELU(),
        dropout_prob=0.6,
        add_skip_connection=True,
        bias=True,
        log_attention_weights=False,
    ):
        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection

        self.linear_proj = nn.Linear(
            num_in_features, num_of_heads * num_out_features, bias=False
        )

        self.scoring_fn_target = nn.Parameter(
            torch.Tensor(1, num_of_heads, num_out_features)
        )
        self.scoring_fn_source = nn.Parameter(
            torch.Tensor(1, num_of_heads, num_out_features)
        )

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter("bias", None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(
                num_in_features, num_of_heads * num_out_features, bias=False
            )
        else:
            self.register_parameter("skip_proj", None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights
        self.attention_weights = None

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def skip_concat_bias(
        self, attention_coefficients, in_nodes_features, out_nodes_features
    ):
        if self.log_attention_weights:
            self.attention_weights = attention_coefficients

        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(
                    -1, self.num_of_heads, self.num_out_features
                )

        if self.concat:
            out_nodes_features = out_nodes_features.view(
                -1, self.num_of_heads * self.num_out_features
            )
        else:
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return (
            out_nodes_features
            if self.activation is None
            else self.activation(out_nodes_features)
        )

    def forward(self, data):
        in_nodes_features, edge_index = data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert (
            edge_index.shape[0] == 2
        ), f"Expected edge index with shape=(2,E) got {edge_index.shape}"

        in_nodes_features = self.dropout(in_nodes_features)

        nodes_features_proj = self.linear_proj(in_nodes_features).view(
            -1, self.num_of_heads, self.num_out_features
        )
        nodes_features_proj = self.dropout(nodes_features_proj)

        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        (
            scores_source_lifted,
            scores_target_lifted,
            nodes_features_proj_lifted,
        ) = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        attentions_per_edge = self.neighborhood_aware_softmax(
            scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes
        )
        attentions_per_edge = self.dropout(attentions_per_edge)

        nodes_features_proj_lifted_weighted = (
            nodes_features_proj_lifted * attentions_per_edge
        )

        out_nodes_features = self.aggregate_neighbors(
            nodes_features_proj_lifted_weighted,
            edge_index,
            in_nodes_features,
            num_of_nodes,
        )

        out_nodes_features = self.skip_concat_bias(
            attentions_per_edge, in_nodes_features, out_nodes_features
        )
        return (out_nodes_features, edge_index)

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()

        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(
            exp_scores_per_edge, trg_index, num_of_nodes
        )

        attentions_per_edge = exp_scores_per_edge / (
            neigborhood_aware_denominator + 1e-16
        )

        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(
        self, exp_scores_per_edge, trg_index, num_of_nodes
    ):
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(
            size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device
        )

        neighborhood_sums.scatter_add_(
            self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge
        )

        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(
        self,
        nodes_features_proj_lifted_weighted,
        edge_index,
        in_nodes_features,
        num_of_nodes,
    ):
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes
        out_nodes_features = torch.zeros(
            size, dtype=in_nodes_features.dtype, device=in_nodes_features.device
        )

        trg_index_broadcasted = self.explicit_broadcast(
            edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted
        )
        out_nodes_features.scatter_add_(
            self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted
        )

        return out_nodes_features

    def lift(
        self, scores_source, scores_target, nodes_features_matrix_proj, edge_index
    ):
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(
            self.nodes_dim, src_nodes_index
        )

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        return this.expand_as(other)


def build_edge_index(adjacency_list_dict, num_of_nodes, device, add_self_edges=True):
    source_nodes_ids, target_nodes_ids = [], []
    seen_edges = set()

    for src_node, neighboring_nodes in adjacency_list_dict.items():
        for trg_node in neighboring_nodes:
            if (src_node, trg_node) not in seen_edges:
                source_nodes_ids.append(src_node)
                target_nodes_ids.append(trg_node)

                seen_edges.add((src_node, trg_node))

    if add_self_edges:
        source_nodes_ids.extend(np.arange(num_of_nodes))
        target_nodes_ids.extend(np.arange(num_of_nodes))

    edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)

    return edge_index


# sanity checking
if __name__ == "__main__":
    # create a random adjacency list
    num_of_nodes = 5
    num_of_neighbors = 2
    num_batches = 2

    # we will treat each batch as a different graph, so we have num_batches disconnected
    # graphs
    adjacency_list_dict = {
        node_id
        + batch
        * num_of_nodes: np.random.choice(
            [
                i
                for i in range(num_of_nodes * batch, num_of_nodes * (batch + 1))
                if i != node_id
            ],
            num_of_neighbors,
            replace=False,
        ).tolist()
        for node_id in range(num_of_nodes)
        for batch in range(num_batches)
    }

    # build the edge index
    edge_index = build_edge_index(adjacency_list_dict, num_of_nodes, device="cpu")

    # create some random node features
    num_of_features_per_node = 8
    in_nodes_features = torch.rand(
        num_batches * num_of_nodes, num_of_features_per_node, dtype=torch.float
    )

    # instantiate a GAT layer
    num_heads = 1
    gat_layer = GATLayer(
        num_in_features=num_of_features_per_node,
        num_out_features=10,
        num_of_heads=num_heads,
        concat=True,
        dropout_prob=0.6,
        add_skip_connection=True,
        bias=True,
        log_attention_weights=True,
    )

    # run it
    out_nodes_features, edge_indexes = gat_layer((in_nodes_features, edge_index))
    print(out_nodes_features.shape)
    print(edge_indexes.shape)
    print(out_nodes_features)
    print(edge_indexes)
