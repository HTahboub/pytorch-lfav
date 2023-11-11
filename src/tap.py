import torch
from torch import nn


class TemporalAttentionPooling(nn.Module):
    """Temporal Attention Pooling (TAP) layer from the LFAV paper.  From the paper:

    "Inputs of the TAP module are audio features and visual features, then outputs
    of the TAP module are video-level audio prediction, video-level visual
    prediction, snippet-level audio prediction and snippet-level visual prediction.
    Video-level predictions are used to train the model, snippet-level predictions
    are used to construct event graphs during training and evaluate model
    performance during validation and testing."

    Args:
        feature_dim: dimension of the input features
        num_classes: number of classes in the dataset
    """

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.sl_fc = nn.Linear(feature_dim, num_classes)
        self.vl_fc = nn.Linear(feature_dim, num_classes)

    def forward(self, video_snippet_embeddings, audio_snippet_embeddings):
        """Forward pass of the TAP layer.

        Args:
            video_snippet_embeddings: (batch_size, num_video_snippets, feature_dim)
            audio_snippet_embeddings: (batch_size, num_audio_snippets, feature_dim)

        Returns:
            vl_video_predictions: (batch_size, num_classes)
            vl_audio_predictions: (batch_size, num_classes)
            sl_video_predictions: (batch_size, num_video_snippets, num_classes)
            sl_audio_predictions: (batch_size, num_audio_snippets, num_classes)
        """
        sl_video_fc_out = self.sl_fc(video_snippet_embeddings)
        sl_audio_fc_out = self.sl_fc(audio_snippet_embeddings)
        sl_video_predictions = torch.sigmoid(sl_video_fc_out)
        sl_audio_predictions = torch.sigmoid(sl_audio_fc_out)

        vl_video_fc_out = self.vl_fc(video_snippet_embeddings)
        vl_audio_fc_out = self.vl_fc(audio_snippet_embeddings)
        video_snippet_weights = torch.softmax(vl_video_fc_out, dim=1)
        audio_snippet_weights = torch.softmax(vl_audio_fc_out, dim=1)

        vl_video_predictions = torch.sum(
            video_snippet_weights * sl_video_predictions, dim=1
        )
        vl_audio_predictions = torch.sum(
            audio_snippet_weights * sl_audio_predictions, dim=1
        )

        return (
            vl_video_predictions,
            vl_audio_predictions,
            sl_video_predictions,
            sl_audio_predictions,
        )
