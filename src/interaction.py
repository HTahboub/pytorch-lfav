import torch
from torch import nn
from torch.nn.functional import cosine_similarity


class EventInteractionModule(nn.Module):
    def __init__(self, feature_dim, num_events, num_heads, dropout, device):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_events = num_events

        self.attention_block = MultimodalAttentionBlock(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            device=device,
        )

    def _reweight_snippets(
        self,
        video_features,
        audio_features,
        video_event_features,
        audio_event_features,
        video_sl_event_predictions,
        audio_sl_event_predictions,
    ):
        """Convert event features to predictions by reweighting snippet-level
        predictions.

        Args:
            video_features: Video features of (batch_size, num_snippets, feature_dim)
            audio_features: Audio features of (batch_size, num_snippets, feature_dim)
            video_event_features: Video event features of
                (batch_size, num_events, feature_dim)
            audio_event_features: Audio event features of
                (batch_size, num_events, feature_dim)
            video_sl_event_predictions: Video snippet-level event predictions of
                (batch_size, num_snippets, num_events)
            audio_sl_event_predictions: Audio snippet-level event predictions of
                (batch_size, num_snippets, num_events)

        Returns:
            video_event_predictions: Video event predictions of
                (batch_size, num_events)
            audio_event_predictions: Audio event predictions of
                (batch_size, num_events)
        """
        video_similarities = cosine_similarity(
            video_features.unsqueeze(2), video_event_features.unsqueeze(1), dim=3
        )
        audio_similarities = cosine_similarity(
            audio_features.unsqueeze(2), audio_event_features.unsqueeze(1), dim=3
        )
        video_weights = torch.softmax(video_similarities, dim=1)
        audio_weights = torch.softmax(audio_similarities, dim=1)

        video_event_predictions = torch.sum(
            video_weights * video_sl_event_predictions, dim=1
        )
        audio_event_predictions = torch.sum(
            audio_weights * audio_sl_event_predictions, dim=1
        )

        return video_event_predictions, audio_event_predictions

    def forward(
        self,
        video_features,
        audio_features,
        video_event_features,
        audio_event_features,
        video_sl_event_predictions,
        audio_sl_event_predictions,
    ):
        """Forward pass of the EventInteraction layer. Takes in event-level features
        and outputs event-level predictions. From the paper:

        "Based on the extracted event features in the second phase, event-aware
        self-attention and cross-modal attention at video-level are performed to explore
        the potential event relations, which are accordingly used to refine the event
        features."

        Args:
            video_features: Video features of (batch_size, num_snippets, feature_dim)
            audio_features: Audio features of (batch_size, num_snippets, feature_dim)
            video_event_features: Video event features of
                (batch_size, num_events, feature_dim)
            audio_event_features: Audio event features of
                (batch_size, num_events, feature_dim)
            video_sl_event_predictions: Video snippet-level event predictions of
                (batch_size, num_snippets, num_events)
            audio_sl_event_predictions: Audio snippet-level event predictions of
                (batch_size, num_snippets, num_events)

        Returns:
            video_event_features: Video event features of
                (batch_size, num_events, feature_dim)
            audio_event_features: Audio event features of
                (batch_size, num_events, feature_dim)
            video_event_predictions: Video event predictions of
                (batch_size, num_events)
            audio_event_predictions: Audio event predictions of
                (batch_size, num_events)
        """
        video_event_features, audio_event_features = self.attention_block(
            video_event_features, audio_event_features
        )

        video_event_predictions, audio_event_predictions = self._reweight_snippets(
            video_features,
            audio_features,
            video_event_features,
            audio_event_features,
            video_sl_event_predictions,
            audio_sl_event_predictions,
        )

        return (
            video_event_features,
            audio_event_features,
            video_event_predictions,
            audio_event_predictions,
        )


class MultimodalAttentionBlock(nn.Module):
    """Multimodal attention block, similar to that in the pyramid transformer, but
    without the pyramid attention masking.

    Args:
        feature_dim: dimension of the event embeddings
        num_heads: number of heads in the multi-head attention layers
    """

    def __init__(
        self,
        feature_dim,
        num_heads,
        device,
        dropout,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.device = device

        self.video_self_fc_q = nn.Linear(feature_dim, feature_dim, device=device)
        self.video_self_fc_k = nn.Linear(feature_dim, feature_dim, device=device)
        self.video_self_fc_v = nn.Linear(feature_dim, feature_dim, device=device)

        self.audio_self_fc_q = nn.Linear(feature_dim, feature_dim, device=device)
        self.audio_self_fc_k = nn.Linear(feature_dim, feature_dim, device=device)
        self.audio_self_fc_v = nn.Linear(feature_dim, feature_dim, device=device)

        self.video_cross_fc_q = nn.Linear(feature_dim, feature_dim, device=device)
        self.video_cross_fc_k = nn.Linear(feature_dim, feature_dim, device=device)
        self.video_cross_fc_v = nn.Linear(feature_dim, feature_dim, device=device)

        self.audio_cross_fc_q = nn.Linear(feature_dim, feature_dim, device=device)
        self.audio_cross_fc_k = nn.Linear(feature_dim, feature_dim, device=device)
        self.audio_cross_fc_v = nn.Linear(feature_dim, feature_dim, device=device)

        self.video_self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, device=device
        )
        self.audio_self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, device=device
        )

        self.video_cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, device=device
        )
        self.audio_cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, device=device
        )

    def forward(self, video_event_embeddings, audio_event_embeddings):
        """Forward pass through the attention block.

        Args:
            video_event_embeddings: (batch_size, num_events, feature_dim)
            audio_event_embeddings: (batch_size, num_events, feature_dim)

        Returns:
            video_event_embeddings: (batch_size, num_events, feature_dim)
            audio_event_embeddings: (batch_size, num_events, feature_dim)
        """
        video_event_embeddings = video_event_embeddings.permute(1, 0, 2)
        audio_event_embeddings = audio_event_embeddings.permute(1, 0, 2)

        # video self-attention
        video_self_q = self.video_self_fc_q(video_event_embeddings)
        video_self_k = self.video_self_fc_k(video_event_embeddings)
        video_self_v = self.video_self_fc_v(video_event_embeddings)

        video_self_attention_output, _ = self.video_self_attention(
            query=video_self_q,
            key=video_self_k,
            value=video_self_v,
        )

        # audio self-attention
        audio_self_q = self.audio_self_fc_q(audio_event_embeddings)
        audio_self_k = self.audio_self_fc_k(audio_event_embeddings)
        audio_self_v = self.audio_self_fc_v(audio_event_embeddings)

        audio_self_attention_output, _ = self.audio_self_attention(
            query=audio_self_q,
            key=audio_self_k,
            value=audio_self_v,
        )

        # video cross-attention
        video_cross_q = self.video_cross_fc_q(video_self_attention_output)
        video_cross_k = self.video_cross_fc_k(audio_self_attention_output)
        video_cross_v = self.video_cross_fc_v(audio_self_attention_output)

        video_cross_attention_output, _ = self.video_cross_attention(
            query=video_cross_q,
            key=video_cross_k,
            value=video_cross_v,
        )

        # audio cross-attention
        audio_cross_q = self.audio_cross_fc_q(audio_self_attention_output)
        audio_cross_k = self.audio_cross_fc_k(video_self_attention_output)
        audio_cross_v = self.audio_cross_fc_v(video_self_attention_output)

        audio_cross_attention_output, _ = self.audio_cross_attention(
            query=audio_cross_q,
            key=audio_cross_k,
            value=audio_cross_v,
        )

        # restore original shape
        video_event_embeddings = video_cross_attention_output.permute(1, 0, 2)
        audio_event_embeddings = audio_cross_attention_output.permute(1, 0, 2)

        return video_event_embeddings, audio_event_embeddings


# testing with dummy data
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = 4
    num_heads = 2
    batch_size = 2
    num_events = 3
    num_snippets = 4
    dropout = 0.2

    video_features = torch.rand((batch_size, num_snippets, feature_dim), device=device)
    audio_features = torch.rand((batch_size, num_snippets, feature_dim), device=device)
    video_event_features = torch.rand(
        (batch_size, num_events, feature_dim), device=device
    )
    audio_event_features = torch.rand(
        (batch_size, num_events, feature_dim), device=device
    )
    video_sl_event_predictions = torch.rand(
        (batch_size, num_snippets, num_events), device=device
    )
    audio_sl_event_predictions = torch.rand(
        (batch_size, num_snippets, num_events), device=device
    )

    event_interaction = EventInteractionModule(
        feature_dim=feature_dim,
        num_events=num_events,
        num_heads=num_heads,
        dropout=dropout,
        device=device,
    )

    video_event_predictions, audio_event_predictions = event_interaction(
        video_features=video_features,
        audio_features=audio_features,
        video_event_features=video_event_features,
        audio_event_features=audio_event_features,
        video_sl_event_predictions=video_sl_event_predictions,
        audio_sl_event_predictions=audio_sl_event_predictions,
    )

    assert video_event_predictions.shape == (batch_size, num_events)
    assert audio_event_predictions.shape == (batch_size, num_events)
