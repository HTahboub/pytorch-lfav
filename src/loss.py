import torch
from torch import nn


class LFAVLoss(nn.Module):
    """Loss function implementation from the LFAV paper."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(
        self,
        s1_vl_video_predictions,
        s1_vl_audio_predictions,
        s2_vl_video_predictions,
        s2_vl_audio_predictions,
        s3_vl_video_predictions,
        s3_vl_audio_predictions,
        vl_video_labels,
        vl_audio_labels,
        video_event_features,
        audio_event_features,
        tap_nn,
    ):
        """Forward pass of the LFAVLoss function.

        Args:
            s1_vl_video_predictions: Video predictions from the first TAP module of
                (batch_size, num_classes)
            s1_vl_audio_predictions: Audio predictions from the first TAP module of
                (batch_size, num_classes)
            s2_vl_video_predictions: Video predictions from the second TAP module of
                (batch_size, num_classes)
            s2_vl_audio_predictions: Audio predictions from the second TAP module of
                (batch_size, num_classes)
            s3_vl_video_predictions: Video predictions from the third TAP module of
                (batch_size, num_classes)
            s3_vl_audio_predictions: Audio predictions from the third TAP module of
                (batch_size, num_classes)
            vl_video_labels: Video labels of (batch_size, num_classes)
            vl_audio_labels: Audio labels of (batch_size, num_classes)
            video_event_features: Video event features of
                (batch_size, num_events, feature_dim)
            audio_event_features: Audio event features of
                (batch_size, num_events, feature_dim)
            tap_nn: torch.nn.Module FC layer that takes in event features and outputs
                event predictions

        Returns:
            total_loss: Total loss of the LFAV model
        """
        batch_size, num_events, _ = video_event_features.shape
        l1_v = self.bce(s1_vl_video_predictions, vl_video_labels)
        l1_a = self.bce(s1_vl_audio_predictions, vl_audio_labels)
        l2_v = self.bce(s2_vl_video_predictions, vl_video_labels)
        l2_a = self.bce(s2_vl_audio_predictions, vl_audio_labels)
        l3_v = self.bce(s3_vl_video_predictions, vl_video_labels)
        l3_a = self.bce(s3_vl_audio_predictions, vl_audio_labels)

        l1 = l1_v + l1_a
        l2 = l2_v + l2_a
        l3 = l3_v + l3_a

        lev, lea = 0, 0
        for event in range(num_events):
            video_event_feature = video_event_features[:, event, :]
            audio_event_feature = audio_event_features[:, event, :]
            video_event_pred = tap_nn(video_event_feature)
            audio_event_pred = tap_nn(audio_event_feature)
            video_event_pred = torch.sigmoid(video_event_pred)[:, event]
            audio_event_pred = torch.sigmoid(audio_event_pred)[:, event]
            lev += self.bce(video_event_pred, vl_video_labels[:, event])
            lea += self.bce(audio_event_pred, vl_audio_labels[:, event])
        lev /= num_events
        lea /= num_events
        le = lev + lea

        total_loss = l1 + l2 + l3 + le
        return total_loss
