import torch
from graph import GraphEventAttentionModule
from interaction import EventInteractionModule
from pyramid import PyramidMultimodalTransformer
from tap import TemporalAttentionPooling
from torch import nn

# TODO: snippets -> 200


class LFAVModel(nn.Module):
    def __init__(
        self,
        device,
        video_dim=1024,
        audio_dim=128,
        feature_dim=512,
        num_pmt_heads=4,
        num_pmt_layers=6,
        pmt_dropout=0.2,
        num_graph_heads=1,
        gat_depth=2,
        graph_dropout=0.2,
        graph_confidence_threshold=0.5,
        num_events=35,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_events = num_events
        self.graph_confidence_threshold = graph_confidence_threshold

        # pre-stage
        self.video_fc = nn.Linear(video_dim, feature_dim)
        self.audio_fc = nn.Linear(audio_dim, feature_dim)

        # used in stages one and two
        # TODO: unsure if these should be shared instead
        self.tap_1 = TemporalAttentionPooling(
            feature_dim=feature_dim,
            num_classes=num_events,
        )
        self.tap_2 = TemporalAttentionPooling(
            feature_dim=feature_dim,
            num_classes=num_events,
        )

        # stage one
        self.pyramid = PyramidMultimodalTransformer(
            feature_dim=feature_dim,
            num_heads=num_pmt_heads,
            num_layers=num_pmt_layers,
            dropout=pmt_dropout,
            device=device,
        )

        # stage two
        self.graph_att = GraphEventAttentionModule(
            feature_dim=feature_dim,
            num_events=num_events,
            num_layers=num_pmt_layers,
            num_heads=num_pmt_heads,
            gat_depth=gat_depth,
            dropout=graph_dropout,
        )

        # stage three
        self.event_interaction = EventInteractionModule(
            feature_dim=feature_dim,
            num_events=num_events,
            num_heads=num_pmt_heads,
            dropout=graph_dropout,
            device=device,
        )

    def forward(self, video_embeddings, audio_embeddings):
        """Forward pass of the LFAV event-centric framework.

        Args:
            video_embeddings: (batch_size, num_video_snippets, video_dim)
            audio_embeddings: (batch_size, num_audio_snippets, audio_dim)

        Returns:
            s1_vl_v_preds: (batch_size, num_video_snippets, num_events)
            s1_vl_a_preds: (batch_size, num_audio_snippets, num_events)
            s1_sl_v_preds: (batch_size, num_events)
            s1_sl_a_preds: (batch_size, num_events)
            s2_vl_v_preds: (batch_size, num_video_snippets, num_events)
            s2_vl_a_preds: (batch_size, num_audio_snippets, num_events)
            s2_sl_v_preds: (batch_size, num_events)
            s2_sl_a_preds: (batch_size, num_events)
            s3_vl_v_preds: (batch_size, num_video_snippets, num_events)
            s3_vl_a_preds: (batch_size, num_audio_snippets, num_events)
        """
        # pre-stage: project video and audio embeddings to feature_dim
        video_embeddings = self.video_fc(video_embeddings)
        audio_embeddings = self.audio_fc(audio_embeddings)

        # stage one: pyramid multimodal transformer
        video_embeddings, audio_embeddings = self.pyramid(
            video_snippet_embeddings=video_embeddings,
            audio_snippet_embeddings=audio_embeddings,
        )
        s1_vl_v_preds, s1_vl_a_preds, s1_sl_v_preds, s1_sl_a_preds = self.tap_1(
            video_snippet_embeddings=video_embeddings,
            audio_snippet_embeddings=audio_embeddings,
        )

        # stage two: graph event attention module
        video_embeddings, audio_embeddings = self.graph_att(
            video_features=video_embeddings,
            audio_features=audio_embeddings,
            video_snippet_preds=s1_sl_v_preds,
            audio_snippet_preds=s1_sl_a_preds,
            confidence_threshold=self.graph_confidence_threshold,
        )
        s2_vl_v_preds, s2_vl_a_preds, s2_sl_v_preds, s2_sl_a_preds = self.tap_2(
            video_snippet_embeddings=video_embeddings,
            audio_snippet_embeddings=audio_embeddings,
        )
        ve_features, ae_features = GraphEventAttentionModule.calculate_event_features(
            video_features=video_embeddings,
            audio_features=audio_embeddings,
            sl_video_predictions=s2_sl_v_preds,
            sl_audio_predictions=s2_sl_a_preds,
            num_events=self.num_events,
        )

        # stage three: event interaction module
        s3_vl_v_preds, s3_vl_a_preds = self.event_interaction(
            video_features=video_embeddings,
            audio_features=audio_embeddings,
            video_event_features=ve_features,
            audio_event_features=ae_features,
            video_sl_event_predictions=s2_sl_v_preds,
            audio_sl_event_predictions=s2_sl_a_preds,
        )

        return (
            s1_vl_v_preds,
            s1_vl_a_preds,
            s1_sl_v_preds,
            s1_sl_a_preds,
            s2_vl_v_preds,
            s2_vl_a_preds,
            s2_sl_v_preds,
            s2_sl_a_preds,
            s3_vl_v_preds,
            s3_vl_a_preds,
        )


if __name__ == "__main__":
    # test small input
    batch_size = 4
    num_snippets = 16
    video_dim = 4
    audio_dim = 2
    feature_dim = 8
    num_pmt_heads = 2
    num_pmt_layers = 2
    pmt_dropout = 0.2
    num_graph_heads = 1
    gat_depth = 2
    graph_dropout = 0.2
    graph_confidence_threshold = 0.5
    num_events = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_embeddings = torch.rand(
        (batch_size, num_snippets, video_dim), device=device, requires_grad=True
    )
    audio_embeddings = torch.rand(
        (batch_size, num_snippets, audio_dim), device=device, requires_grad=True
    )

    model = LFAVModel(
        device=device,
        video_dim=video_dim,
        audio_dim=audio_dim,
        feature_dim=feature_dim,
        num_pmt_heads=num_pmt_heads,
        num_pmt_layers=num_pmt_layers,
        pmt_dropout=pmt_dropout,
        num_graph_heads=num_graph_heads,
        gat_depth=gat_depth,
        graph_dropout=graph_dropout,
        graph_confidence_threshold=graph_confidence_threshold,
        num_events=num_events,
    )
    preds = model(video_embeddings, audio_embeddings)

    # fmt: off
    # test shapes
    print(*(pred.shape for pred in preds if pred is not None), sep="\n")
    assert all(pred.shape == (batch_size, num_events) for pred in preds[:2])
    assert all(pred.shape == (batch_size, num_snippets, num_events) for pred in preds[2:4])  # noqa: E501
    assert all(pred.shape == (batch_size, num_events) for pred in preds[4:6])
    assert all(pred.shape == (batch_size, num_snippets, num_events) for pred in preds[6:8])  # noqa: E501
    assert all(pred.shape == (batch_size, num_events) for pred in preds[8:])
    # fmt: on

    # test diffentiability
    loss = preds[-1].sum() + preds[-2].sum()
    loss.backward()
    if torch.all(video_embeddings.grad == 0) or torch.all(audio_embeddings.grad == 0):
        if torch.all(video_embeddings.grad == 0):
            print("video_embeddings.grad is all zeros")
        if torch.all(audio_embeddings.grad == 0):
            print("audio_embeddings.grad is all zeros")
    elif video_embeddings.grad is None or audio_embeddings.grad is None:
        if video_embeddings.grad is None:
            print("Not differentiable w.r.t. video_embeddings")
        if audio_embeddings.grad is None:
            print("Not differentiable w.r.t. audio_embeddings")
    else:
        print("Differentiable w.r.t. video_embeddings and audio_embeddings")
