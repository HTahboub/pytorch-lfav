import torch
from torch import nn
from torchview import draw_graph
import torchviz

from utils.mask import create_attention_mask


class PyramidMultimodalTransformer(nn.Module):
    """Implementation of the Pyramid Multimodal Transformer (PMT) from the LFAV paper.
    Represents stage one of the overall architecture.

    Args:
        feature_dim: dimension of the video and audio embeddings
        num_heads: number of heads in the multi-head attention layers
        num_layers: number of layers in the transformer
        dropout: dropout probability
        device: torch device
    """

    def __init__(self, feature_dim, num_heads, num_layers, dropout, device):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.layers = []
        for i in range(1, num_layers + 1):
            self.layers.append(
                PyramidMultimodalTransformerLayer(
                    feature_dim=feature_dim,
                    num_heads=num_heads,
                    layer_num=i,
                    dropout=dropout,
                    device=self.device,
                )
            )
        self.layers = nn.ModuleList(self.layers)

    def forward(self, video_snippet_embeddings, audio_snippet_embeddings):
        """Forward pass through the model.

        Args:
            video_snippet_embeddings: (batch_size, num_snippets, feature_dim)
            audio_snippet_embeddings: (batch_size, num_snippets, feature_dim)

        Returns:
            video_snippet_embeddings: (batch_size, num_snippets, feature_dim)
            audio_snippet_embeddings: (batch_size, num_snippets, feature_dim)
        """
        for layer in self.layers:
            video_snippet_embeddings, audio_snippet_embeddings = layer(
                video_snippet_embeddings, audio_snippet_embeddings
            )
        return video_snippet_embeddings, audio_snippet_embeddings


class PyramidMultimodalTransformerLayer(nn.Module):
    """Implementation of a single layer of the Pyramid Multimodal Transformer (PMT)
    from the LFAV paper.

    Args:
        feature_dim: dimension of the video and audio embeddings
        num_heads: number of heads in the multi-head attention layers
        layer_num: the layer number (1-indexed)
        dropout: dropout probability
        device: torch device
    """

    def __init__(self, feature_dim, num_heads, layer_num, dropout, device):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.layer_num = layer_num
        self.dropout = dropout
        self.device = device

        self.attention_one = MultimodalAttentionBlock(
            feature_dim=feature_dim,
            num_heads=num_heads,
            layer_num=layer_num,
            dropout=dropout,
            device=device,
        )
        self.attention_two = MultimodalAttentionBlock(
            feature_dim=feature_dim,
            num_heads=num_heads,
            layer_num=layer_num,
            dropout=dropout,
            device=device,
        )

    def forward(self, video_snippet_embeddings, audio_snippet_embeddings):
        """Forward pass through the layer.

        Includes the snippet shift operation. From the paper:

        "To avoid the potential partition of an event caused by the pyramid window
        module, we also propose a kind of snippet shift strategy to capture the events
        across different pyramid windows. As shown in Fig. 4, we shift the top 2^l/2
        snippet features in the first window to the end of the video after the first
        multimodal attention module, then feed the updated snippet sequence to the
        second attention module. At the end of the PMT layer, we restore the temporal
        order of snippets and obtain the updated snippet features."

        Args:
            video_snippet_embeddings: (batch_size, num_snippets, feature_dim)
            audio_snippet_embeddings: (batch_size, num_snippets, feature_dim)

        Returns:
            video_snippet_embeddings: (batch_size, num_snippets, feature_dim)
            audio_snippet_embeddings: (batch_size, num_snippets, feature_dim)
        """

        # apply first multimodal attention block
        video_snippet_embeddings, audio_snippet_embeddings = self.attention_one(
            video_snippet_embeddings, audio_snippet_embeddings
        )

        # snippet shift
        shift_size = (2**self.layer_num) // 2
        video_snippet_embeddings = torch.roll(
            video_snippet_embeddings, -shift_size, dims=2
        )

        # apply second multimodal attention block
        video_snippet_embeddings, audio_snippet_embeddings = self.attention_two(
            video_snippet_embeddings, audio_snippet_embeddings
        )

        # inverse snippet shift
        video_snippet_embeddings = torch.roll(
            video_snippet_embeddings, shift_size, dims=2
        )

        return video_snippet_embeddings, audio_snippet_embeddings


class MultimodalAttentionBlock(nn.Module):
    """Implementation of the multimodal attention block from the LFAV paper.

    Args:
        feature_dim: dimension of the video and audio embeddings
        num_heads: number of heads in the multi-head attention layers
        layer_num: the layer number (1-indexed)
    """

    def __init__(
        self,
        feature_dim,
        num_heads,
        layer_num,
        device,
        dropout,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.layer_num = layer_num
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

    def forward(self, video_snippet_embeddings, audio_snippet_embeddings):
        """Forward pass through the attention block.

        Four attention operations. From the paper:

        "Each multimodal attention module contains four attention units, two of them
        are for audio and visual modeling, while the other two are for crossmodal
        interaction at the snippet level. Then, the updated snippet features are used
        as the inputs for the next layer."

        Args:
            video_snippet_embeddings: (batch_size, num_snippets, feature_dim)
            audio_snippet_embeddings: (batch_size, num_snippets, feature_dim)

        Returns:
            video_snippet_embeddings: (batch_size, num_snippets, feature_dim)
            audio_snippet_embeddings: (batch_size, num_snippets, feature_dim)
        """
        video_snippet_embeddings = video_snippet_embeddings.permute(1, 0, 2)
        audio_snippet_embeddings = audio_snippet_embeddings.permute(1, 0, 2)

        video_attn_mask = create_attention_mask(
            len(video_snippet_embeddings), self.layer_num, self.device
        )
        audio_attn_mask = create_attention_mask(
            len(audio_snippet_embeddings), self.layer_num, self.device
        )

        # video self-attention
        video_self_q = self.video_self_fc_q(video_snippet_embeddings)
        video_self_k = self.video_self_fc_k(video_snippet_embeddings)
        video_self_v = self.video_self_fc_v(video_snippet_embeddings)

        video_self_attention_output, _ = self.video_self_attention(
            query=video_self_q,
            key=video_self_k,
            value=video_self_v,
            attn_mask=video_attn_mask,
        )

        # audio self-attention
        audio_self_q = self.audio_self_fc_q(audio_snippet_embeddings)
        audio_self_k = self.audio_self_fc_k(audio_snippet_embeddings)
        audio_self_v = self.audio_self_fc_v(audio_snippet_embeddings)

        audio_self_attention_output, _ = self.audio_self_attention(
            query=audio_self_q,
            key=audio_self_k,
            value=audio_self_v,
            attn_mask=audio_attn_mask,
        )

        # video cross-attention
        video_cross_q = self.video_cross_fc_q(video_self_attention_output)
        video_cross_k = self.video_cross_fc_k(audio_self_attention_output)
        video_cross_v = self.video_cross_fc_v(audio_self_attention_output)

        video_cross_attention_output, _ = self.video_cross_attention(
            query=video_cross_q,
            key=video_cross_k,
            value=video_cross_v,
            attn_mask=video_attn_mask,
        )

        # audio cross-attention
        audio_cross_q = self.audio_cross_fc_q(audio_self_attention_output)
        audio_cross_k = self.audio_cross_fc_k(video_self_attention_output)
        audio_cross_v = self.audio_cross_fc_v(video_self_attention_output)

        audio_cross_attention_output, _ = self.audio_cross_attention(
            query=audio_cross_q,
            key=audio_cross_k,
            value=audio_cross_v,
            attn_mask=audio_attn_mask,
        )

        # restore original shape
        video_snippet_embeddings = video_cross_attention_output.permute(1, 0, 2)
        audio_snippet_embeddings = audio_cross_attention_output.permute(1, 0, 2)

        return video_snippet_embeddings, audio_snippet_embeddings


##################### TESTING #####################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_video_snippet_embeddings = torch.rand(
        (16, 32, 512), device=device, requires_grad=True
    )
    dummy_audio_snippet_embeddings = torch.rand(
        (16, 32, 512), device=device, requires_grad=True
    )

    model = PyramidMultimodalTransformer(feature_dim=512, num_heads=4, num_layers=6)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    video_snippet_embeddings, audio_snippet_embeddings = model(
        dummy_video_snippet_embeddings, dummy_audio_snippet_embeddings
    )
    assert video_snippet_embeddings.shape == dummy_video_snippet_embeddings.shape
    assert audio_snippet_embeddings.shape == dummy_audio_snippet_embeddings.shape
    assert torch.any(video_snippet_embeddings != dummy_video_snippet_embeddings)
    assert torch.any(audio_snippet_embeddings != dummy_audio_snippet_embeddings)

    # check differentiability
    (video_snippet_embeddings.sum() + audio_snippet_embeddings.sum()).backward()
    assert dummy_audio_snippet_embeddings.grad is not None, "Not differentiable"
    assert dummy_video_snippet_embeddings.grad is not None, "Not differentiable"
    print("Checks passed!")

    # generate figures for sanity check
    torchviz.make_dot(  # backward computation graph
        (video_snippet_embeddings.sum(), audio_snippet_embeddings.sum()),
        params=dict(model.named_parameters()),
    ).render("misc/torchviz", format="png")
    draw_graph(  # forward computation graph
        model,
        input_data=(dummy_video_snippet_embeddings, dummy_audio_snippet_embeddings),
        save_graph=True,
        directory="misc/torchview",
    )
    print("Figures saved!")
