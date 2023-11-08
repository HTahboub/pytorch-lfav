from torch import nn

from pyramid import PyramidMultimodalTransformer
from tap import TemporalAttentionPooling


class LFAVModel(nn.Module):
    def __init__(
        self,
        video_dim=1024,
        audio_dim=128,
        feature_dim=512,
        num_pmt_heads=4,
        num_pmt_layers=6,
        num_classes=35,
    ):
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # pre-stage
        self.video_fc = nn.Linear(video_dim, feature_dim)
        self.audio_fc = nn.Linear(audio_dim, feature_dim)

        # used in stages one and two
        self.tap = TemporalAttentionPooling(
            feature_dim=feature_dim,
            num_classes=num_classes,
        )

        # stage one
        self.pyramid = PyramidMultimodalTransformer(
            feature_dim=feature_dim, num_heads=num_pmt_heads, num_layers=num_pmt_layers
        )

        # stage two
        # TODO

        # stage three
        # TODO

    def forward(self, video_embeddings, audio_embeddings):
        """Forward pass of the LFAV event-centric framework.

        Args:
            video_embeddings: (batch_size, num_video_snippets, video_dim)
            audio_embeddings: (batch_size, num_audio_snippets, audio_dim)

        Returns:
            vl_video_predictions: (batch_size, num_classes)
            vl_audio_predictions: (batch_size, num_classes)
            sl_video_predictions: (batch_size, num_video_snippets, num_classes)
            sl_audio_predictions: (batch_size, num_audio_snippets, num_classes)
        """
        pass
