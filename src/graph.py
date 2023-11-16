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
            for _ in range(num_layers)
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
            for _ in range(num_layers)
        ]
        self.video_modules = nn.ModuleList(self.video_modules)

    @abstractmethod
    def _build_adjacency(snippet_preds, confidence_threshold):
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
            Dict[int, List[int]]: Adjacency dictionary, where the keys are node indices
                and the values are lists of adjacent node indices. Represents a set of
                batch_size disconnected graphs, one for each video (snippet sequence).
        """
        batch_size, num_snippets = snippet_preds.shape
        adj_dict = {node: [] for node in range(batch_size * num_snippets)}
        # we will make batch_size disconnected graphs
        for i in range(snippet_preds.shape[0]):
            # temporal edges
            for j in range(num_snippets * i, num_snippets * (i + 1) - 1):
                adj_dict[j].append(j + 1)
                adj_dict[j + 1].append(j)
            # semantic edges
            nodes = []
            for j in range(num_snippets * i, num_snippets * (i + 1)):
                if snippet_preds[i, j % num_snippets] >= confidence_threshold:
                    nodes.append(j)
            for node in nodes:
                for other_node in nodes:
                    if node != other_node and other_node not in adj_dict[node]:
                        adj_dict[node].append(other_node)
        return adj_dict

    @staticmethod
    def calculate_event_features(
        video_features,
        audio_features,
        sl_video_predictions,
        sl_audio_predictions,
        num_events,
    ):
        """Calculate event features from snippet-level predictions and audio/video
        features. From the paper:

        "Based on the refined event-aware snippet features, we employ the TAP module to
        obtain their importance (i.e., attention weight) of the specific event, then
        perform weighted aggregation over the snippet features belonging to the same
        event to generate the final event feature."

        Args:
            video_features: Video features of (batch_size, num_snippets, feature_dim)
            audio_features: Audio features of (batch_size, num_snippets, feature_dim)
            sl_video_predictions: Video snippet-level predictions of
                (batch_size, num_snippets, num_events)
            sl_audio_predictions: Audio snippet-level predictions of
                (batch_size, num_snippets, num_events)

        Returns:
            video_event_features: Video event features of
                (batch_size, num_events, feature_dim)
            audio_event_features: Audio event features of
                (batch_size, num_events, feature_dim)
        """
        batch_size, num_snippets, feature_dim = video_features.shape

        normalized_weights = torch.softmax(sl_video_predictions, dim=2)
        video_event_features = torch.sum(
            video_features.unsqueeze(2) * normalized_weights.unsqueeze(3), dim=1
        )

        normalized_weights = torch.softmax(sl_audio_predictions, dim=2)
        audio_event_features = torch.sum(
            audio_features.unsqueeze(2) * normalized_weights.unsqueeze(3), dim=1
        )

        return video_event_features, audio_event_features

    def forward(
        self,
        video_features,
        audio_features,
        video_snippet_preds,
        audio_snippet_preds,
        confidence_threshold,
    ):
        """Forward pass of the GraphEventAttentionModule. Takes in audio and video
        features and snippet-level predictions, and outputs event-level predictions.

        Args:
            video_features: Video features of (batch_size, num_snippets, feature_dim)
            audio_features: Audio features of (batch_size, num_snippets, feature_dim)
            video_snippet_preds: Video snippet-level predictions of
                (batch_size, num_snippets, num_events)
            audio_snippet_preds: Audio snippet-level predictions of
                (batch_size, num_snippets, num_events)
            confidence_threshold: Confidence threshold for semantic edges.

        Returns:
            video_features: Video features of (batch_size, num_snippets, feature_dim)
            audio_features: Audio features of (batch_size, num_snippets, feature_dim)
        """
        batch_size, num_snippets, feature_dim = audio_features.shape
        audio_event_adj = []
        video_event_adj = []
        for i in range(self.num_events):
            audio_adj = GraphEventAttentionModule._build_adjacency(
                audio_snippet_preds[:, :, i], confidence_threshold
            )
            video_adj = GraphEventAttentionModule._build_adjacency(
                video_snippet_preds[:, :, i], confidence_threshold
            )
            audio_adj = build_edge_index(
                audio_adj,
                num_of_nodes=num_snippets * batch_size,
                device=audio_features.device,
            )
            video_adj = build_edge_index(
                video_adj,
                num_of_nodes=num_snippets * batch_size,
                device=video_features.device,
            )
            audio_event_adj.append(audio_adj)
            video_event_adj.append(video_adj)

        # audio
        for module in self.audio_modules:
            aggregated = torch.zeros_like(audio_features)
            for i in range(self.num_events):
                gat_features, _ = module(
                    (audio_features.reshape(-1, self.feature_dim), audio_event_adj[i])
                )
                aggregated += gat_features.view(*audio_features.shape)
            audio_features = aggregated / self.num_events

        # video
        for module in self.video_modules:
            aggregated = torch.zeros_like(video_features)
            for i in range(self.num_events):
                gat_features, _ = module(
                    (video_features.reshape(-1, self.feature_dim), video_event_adj[i])
                )
                aggregated += gat_features.view(*video_features.shape)
            video_features = aggregated / self.num_events

        return video_features, audio_features


if __name__ == "__main__":
    # sanity check _build_adjacency
    snippet_preds = torch.tensor(
        [
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            [0.1, 0.3, 0.3, 0.6, 0.5, 0.4],
            [0.1, 0.8, 0.1, 0.6, 0.5, 0.4],
        ],
        dtype=torch.float,
    )
    confidence_threshold = 0.5
    adj_dict = GraphEventAttentionModule._build_adjacency(
        snippet_preds, confidence_threshold
    )
    print(adj_dict, sep="\n")

    # sanity check forward
    feature_dim = 6
    num_events = 3
    num_layers = 2
    num_heads = 1
    gat_depth = 2
    dropout = 0.0
    num_batches = 3
    num_snippets = 6
    confidence_threshold = 0.5

    torch.manual_seed(42)
    audio_features = torch.rand((num_batches, num_snippets, feature_dim))
    video_features = torch.rand((num_batches, num_snippets, feature_dim))
    video_snippet_preds = torch.rand((num_batches, num_snippets, num_events))
    audio_snippet_preds = torch.rand((num_batches, num_snippets, num_events))
    module = GraphEventAttentionModule(
        feature_dim, num_events, num_layers, num_heads, gat_depth, dropout
    )
    audio_features_out, video_features_out = module(
        video_features,
        audio_features,
        video_snippet_preds,
        audio_snippet_preds,
        confidence_threshold,
    )
    print("input audio shape:", audio_features.shape)
    print("input video shape:", video_features.shape)
    print("output audio shape:", audio_features_out.shape)
    print("output video shape:", video_features_out.shape)
