import torch
from torch import nn


class LFAVLoss(nn.Module):
    """Loss function implementation from the LFAV paper."""

    def __init__(self, event_weight=0.3):
        super().__init__()
        self.bce = nn.BCELoss()
        self.event_weight = event_weight

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
            f1_video: F1 score of the video predictions
            f1_audio: F1 score of the audio predictions
            f1_av: F1 score of the audio-visual predictions
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

        total_loss = (1 - self.event_weight) * (l1 + l2 + l3) + self.event_weight * le

        # audio-visual version: turn on preds and labels from both
        av_preds = torch.logical_or(
            torch.round(s2_vl_video_predictions), torch.round(s2_vl_audio_predictions)
        ).float()
        av_labels = torch.logical_or(vl_video_labels, vl_audio_labels).float()

        # compute metrics required for F1 scores
        tp_video = torch.sum(
            torch.round(s2_vl_video_predictions) * vl_video_labels
        ).float()
        tp_audio = torch.sum(
            torch.round(s2_vl_audio_predictions) * vl_audio_labels
        ).float()
        tp_av = torch.sum(av_preds * av_labels).float()

        fp_video = torch.sum(
            torch.round(s2_vl_video_predictions) * (1 - vl_video_labels)
        ).float()
        fp_audio = torch.sum(
            torch.round(s2_vl_audio_predictions) * (1 - vl_audio_labels)
        ).float()
        fp_av = torch.sum(av_preds * (1 - av_labels)).float()

        fn_video = torch.sum(
            (1 - torch.round(s2_vl_video_predictions)) * vl_video_labels
        ).float()
        fn_audio = torch.sum(
            (1 - torch.round(s2_vl_audio_predictions)) * vl_audio_labels
        ).float()
        fn_av = torch.sum((1 - av_preds) * av_labels).float()

        precision_video = tp_video / (tp_video + fp_video)
        precision_audio = tp_audio / (tp_audio + fp_audio)
        precision_av = tp_av / (tp_av + fp_av)

        recall_video = tp_video / (tp_video + fn_video)
        recall_audio = tp_audio / (tp_audio + fn_audio)
        recall_av = tp_av / (tp_av + fn_av)

        # final F1 scores for each component
        f1_video = 2 * (precision_video * recall_video) / (
            precision_video + recall_video
        )
        f1_audio = 2 * (precision_audio * recall_audio) / (
            precision_audio + recall_audio
        )
        f1_av = 2 * (precision_av * recall_av) / (precision_av + recall_av)

        return total_loss, f1_video, f1_audio, f1_av
