from abc import abstractmethod

import torch
from torch import nn
from gat import GraphAttentionNetwork, build_edge_index


class GraphEventAttentionModule(nn.Module):
    def __init__(
        self, feature_dim, num_events, num_layers, num_heads, gat_depth, dropout
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_events = num_events
        self.num_layers = num_layers

        audio_modules = [
            GraphAttentionNetwork(
                num_of_layers=gat_depth,
                num_heads_per_layer=[num_heads] * gat_depth,
                num_features_per_layer=[feature_dim] * (gat_depth + 1),
                add_skip_connection=True,
                bias=True,
                dropout=dropout,
                log_attention_weights=False,
            )
            for _ in num_layers
        ]
        self.audio_modules = nn.ModuleList(audio_modules)

        self.video_modules = [
            GraphAttentionNetwork(
                num_of_layers=gat_depth,
                num_heads_per_layer=[num_heads] * gat_depth,
                num_features_per_layer=[feature_dim] * (gat_depth + 1),
                add_skip_connection=True,
                bias=True,
                dropout=dropout,
                log_attention_weights=False,
            )
            for _ in num_layers
        ]
        self.video_modules = nn.ModuleList(self.video_modules)

    @abstractmethod
    def build_adjacency(snippet_preds, confidence_threshold):
        """Build adjacency dictionary from snippet pred confidences. From the paper:

        "We construct a graph for each event category of each modality, where each
        snippet is a node of the graph and the nodes are connected according to two
        kinds of edges: temporal edges and semantic edges. For temporal edges, we
        connect every two adjacent nodes, because successive snippets are usually
        considered to be similar in semantics. For semantic edges, the nodes are fully
        connected when they all have high confidence in the same event category. The
        confidence comes from the snippet-level predictions in the first phase."

        Args:
            snippet_preds: Snippet-level predictions of shape (batch_size, num_snippets)
            confidence_threshold: Confidence threshold for semantic edges.

        Returns:
            List[Dict[int, List[int]]]: List of adjacency dictionaries, one for each
                batch.
        """
        adj_dicts = []
        for i in range(snippet_preds.shape[0]):
            adj_dict = {node: [] for node in range(snippet_preds[i].shape[0])}
            # temporal edges
            for j in range(snippet_preds.shape[1] - 1):
                adj_dict[j].append(j + 1)
                adj_dict[j + 1].append(j)
            # semantic edges
            nodes = []
            for j in range(snippet_preds.shape[1]):
                if snippet_preds[i, j] >= confidence_threshold:
                    nodes.append(j)
            for node in nodes:
                for other_node in nodes:
                    if node != other_node and other_node not in adj_dict[node]:
                        adj_dict[node].append(other_node)
            adj_dicts.append(adj_dict)
        return adj_dicts

    def forward(
        self, audio_features, video_features, video_snippet_preds, audio_snippet_preds
    ):
        audio_adj = []
        video_adj = []
        for i in range(self.num_events):
            audio_adj.append(
                self.build_adjacency(audio_features, audio_snippet_preds[:, :, i])
            )
            video_adj.append(
                self.build_adjacency(video_features, video_snippet_preds[:, :, i])
            )


if __name__ == "__main__":
    # test build_adjacency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snippet_preds = torch.tensor(
        [
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            [0.1, 0.3, 0.3, 0.6, 0.5, 0.4],
            [0.1, 0.8, 0.1, 0.6, 0.5, 0.4],
        ],
        dtype=torch.float,
        requires_grad=True,
        device=device,
    )
    confidence_threshold = 0.5
    adj_dicts = GraphEventAttentionModule.build_adjacency(
        snippet_preds, confidence_threshold
    )
    print(*adj_dicts, sep="\n")
