import os
import torch
import wandb
from graph import GraphEventAttentionModule
from interaction import EventInteractionModule
from pyramid import PyramidMultimodalTransformer
from tap import TemporalAttentionPooling
from torch import nn


CHECKPOINT_PATH = "/scratch/tahboub.h/checkpoints/lfav/"


class LFAVModel(nn.Module):
    def __init__(
        self,
        device,
        overfit=False,
        video_dim=1024,
        audio_dim=128,
        feature_dim=512,
        num_pmt_heads=4,
        num_pmt_layers=6,
        pmt_dropout=0.2,
        num_graph_heads=1,
        gat_depth=2,
        graph_dropout=0.0,
        event_dropout=0.0,
        graph_confidence_threshold=0.51,
        num_events=35,
    ):
        super().__init__()

        # save hyperparams
        ignore = ["self", "device", "ignore"]
        for k, v in locals().items():
            if k not in ignore:
                wandb.config[k] = v

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
            dropout=pmt_dropout if not overfit else 0,
            device=device,
        )

        # stage two
        self.graph_att = GraphEventAttentionModule(
            feature_dim=feature_dim,
            num_events=num_events,
            num_layers=num_pmt_layers,
            num_heads=num_graph_heads,
            gat_depth=gat_depth,
            dropout=graph_dropout if not overfit else 0,
        )

        # stage three
        self.event_interaction = EventInteractionModule(
            feature_dim=feature_dim,
            num_events=num_events,
            num_heads=num_pmt_heads,
            dropout=event_dropout if not overfit else 0,
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
        s1_vl_v_preds, s1_vl_a_preds, s1_sl_v_preds, s1_sl_a_preds, _ = self.tap_1(
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
        s2_vl_v_preds, s2_vl_a_preds, s2_sl_v_preds, s2_sl_a_preds, tap_nn = self.tap_2(
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
        s3out = self.event_interaction(
            video_features=video_embeddings,
            audio_features=audio_embeddings,
            video_event_features=ve_features,
            audio_event_features=ae_features,
            video_sl_event_predictions=s2_sl_v_preds,
            audio_sl_event_predictions=s2_sl_a_preds,
        )
        video_event_features, audio_event_features, s3_vl_v_preds, s3_vl_a_preds = s3out

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
            video_event_features,
            audio_event_features,
            tap_nn,  # hacky
        )


def train(
    model,
    optimizer,
    scheduler,
    criterion,
    train_dataloader,
    val_dataloader,
    num_epochs,
    experiment,
    device,
):
    """Train the model.

    Args:
        model: LFAVModel
        optimizer: torch.optim.Optimizer
        scheduler: torch.optim.lr_scheduler
        criterion: torch.nn.Module
        train_dataloader: torch.utils.data.DataLoader
        val_dataloader: torch.utils.data.DataLoader | None
        num_epochs: int
        experiment: str
        device: torch.device

    Returns:
        model: LFAVModel
    """
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_dataloader,
            epoch=epoch,
            total_epochs=num_epochs,
            device=device,
        )
        scheduler.step()
        if val_dataloader is not None:
            val_loss = evaluate(
                model=model,
                criterion=criterion,
                dataloader=val_dataloader,
                device=device,
                epoch=epoch,
            )
        print(f"Train loss: {train_loss:.4f}")
        if val_dataloader is not None:
            print(f"Validation loss: {val_loss:.4f}")
            print()

        # save checkpoint
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            ckpt_path = os.path.join(CHECKPOINT_PATH, experiment)
            os.makedirs(ckpt_path, exist_ok=True)
            ckpt_path = os.path.join(ckpt_path, str(len(os.listdir(ckpt_path))))
            os.makedirs(ckpt_path, exist_ok=False)

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss if val_dataloader is not None else None,
            }
            torch.save(ckpt, os.path.join(ckpt_path, f"epoch_{epoch}.ckpt"))

    return model


def train_one_epoch(
    model, optimizer, criterion, dataloader, epoch, total_epochs, device
):
    """Train the model for one epoch.

    Args:
        model: LFAVModel
        optimizer: torch.optim.Optimizer
        criterion: torch.nn.Module
        dataloader: torch.utils.data.DataLoader
        epoch: int
        total_epochs: int
        device: torch.device

    Returns:
        total_loss: float
    """
    model.train()
    total_loss = 0
    total_f1_video = total_f1_audio = total_f1_av = 0
    for i, (
        video_embeddings,
        audio_embeddings,
        video_labels,
        audio_labels,
    ) in enumerate(dataloader):
        video_embeddings = video_embeddings.to(device)
        audio_embeddings = audio_embeddings.to(device)
        video_labels = video_labels.to(device)
        audio_labels = audio_labels.to(device)
        optimizer.zero_grad()
        preds = model(video_embeddings, audio_embeddings)
        (
            s1_vl_v_preds,
            s1_vl_a_preds,
            _,
            _,
            s2_vl_v_preds,
            s2_vl_a_preds,
            _,
            _,
            s3_vl_v_preds,
            s3_vl_a_preds,
            video_event_features,
            audio_event_features,
            tap_nn,
        ) = preds



        s1_vl_v_preds = torch.sigmoid(s1_vl_v_preds)
        s1_vl_a_preds = torch.sigmoid(s1_vl_a_preds)
        s2_vl_v_preds = torch.sigmoid(s2_vl_v_preds)
        s2_vl_a_preds = torch.sigmoid(s2_vl_a_preds)
        s3_vl_v_preds = torch.sigmoid(s3_vl_v_preds)
        s3_vl_a_preds = torch.sigmoid(s3_vl_a_preds)

        loss, f1_video, f1_audio, f1_av = criterion(
            s1_vl_video_predictions=s1_vl_v_preds,
            s1_vl_audio_predictions=s1_vl_a_preds,
            s2_vl_video_predictions=s2_vl_v_preds,
            s2_vl_audio_predictions=s2_vl_a_preds,
            s3_vl_video_predictions=s3_vl_v_preds,
            s3_vl_audio_predictions=s3_vl_a_preds,
            vl_video_labels=video_labels,
            vl_audio_labels=audio_labels,
            video_event_features=video_event_features,
            audio_event_features=audio_event_features,
            tap_nn=tap_nn,
        )
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        total_f1_video += f1_video
        total_f1_audio += f1_audio
        total_f1_av += f1_av

        if i % 10 == 0 and len(dataloader) > 10:
            wandb.log(
                {
                    "batch_train_loss": batch_loss,
                    "batch_train_f1_video": f1_video,
                    "batch_train_f1_audio": f1_audio,
                    "batch_train_f1_av": f1_av,
                    "epoch": epoch,
                    "step": i + epoch * len(dataloader),
                }
            )

    total_loss /= len(dataloader)
    total_f1_video /= len(dataloader)
    total_f1_audio /= len(dataloader)
    total_f1_av /= len(dataloader)
    wandb.log(
        {
            "train_loss": total_loss,
            "train_f1_video": total_f1_video,
            "train_f1_audio": total_f1_audio,
            "train_f1_av": total_f1_av,
            "epoch": epoch,
        }
    )
    return total_loss


def evaluate(model, criterion, dataloader, device, epoch=None):
    """Evaluate the model.

    Args:
        model: LFAVModel
        criterion: torch.nn.Module
        dataloader: torch.utils.data.DataLoader
        device: torch.device

    Returns:
        total_loss: float
    """
    model.eval()
    total_loss = 0
    total_f1_video = total_f1_audio = total_f1_av = 0
    with torch.no_grad():
        for i, (
            video_embeddings,
            audio_embeddings,
            video_labels,
            audio_labels,
        ) in enumerate(dataloader):
            video_embeddings = video_embeddings.to(device)
            audio_embeddings = audio_embeddings.to(device)
            video_labels = video_labels.to(device)
            audio_labels = audio_labels.to(device)

            preds = model(video_embeddings, audio_embeddings)
            (
                s1_vl_v_preds,
                s1_vl_a_preds,
                _,
                _,
                s2_vl_v_preds,
                s2_vl_a_preds,
                _,
                _,
                s3_vl_v_preds,
                s3_vl_a_preds,
                video_event_features,
                audio_event_features,
                tap_nn,
            ) = preds
            loss, f1_video, f1_audio, f1_av = criterion(
                s1_vl_video_predictions=s1_vl_v_preds,
                s1_vl_audio_predictions=s1_vl_a_preds,
                s2_vl_video_predictions=s2_vl_v_preds,
                s2_vl_audio_predictions=s2_vl_a_preds,
                s3_vl_video_predictions=s3_vl_v_preds,
                s3_vl_audio_predictions=s3_vl_a_preds,
                vl_video_labels=video_labels,
                vl_audio_labels=audio_labels,
                video_event_features=video_event_features,
                audio_event_features=audio_event_features,
                tap_nn=tap_nn,
            )
            total_loss += loss.item()
            total_f1_video += f1_video
            total_f1_audio += f1_audio
            total_f1_av += f1_av
    total_loss /= len(dataloader)
    total_f1_video /= len(dataloader)
    total_f1_audio /= len(dataloader)
    total_f1_av /= len(dataloader)
    if epoch is not None:
        wandb.log(
            {
                "val_loss": total_loss,
                "val_f1_video": total_f1_video,
                "val_f1_audio": total_f1_audio,
                "val_f1_av": total_f1_av,
                "epoch": epoch,
            }
        )
    return total_loss


if __name__ == "__main__":
    # test small input
    batch_size = 4
    num_batches = 5
    num_videos = batch_size * num_batches
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
    assert all(pred.shape == (batch_size, num_events) for pred in preds[:2])
    assert all(pred.shape == (batch_size, num_snippets, num_events) for pred in preds[2:4])  # noqa: E501
    assert all(pred.shape == (batch_size, num_events) for pred in preds[4:6])
    assert all(pred.shape == (batch_size, num_snippets, num_events) for pred in preds[6:8])  # noqa: E501
    assert all(pred.shape == (batch_size, num_events) for pred in preds[8:10])
    # fmt: on

    # test diffentiability
    loss = preds[-4].sum() + preds[-5].sum()
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

    # test train
    from loss import LFAVLoss
    from torch import optim
    from torch.utils.data import DataLoader, TensorDataset

    video_embeddings = torch.rand(
        (num_videos, num_snippets, video_dim), device=device, requires_grad=True
    )
    audio_embeddings = torch.rand(
        (num_videos, num_snippets, audio_dim), device=device, requires_grad=True
    )
    dataset = TensorDataset(
        video_embeddings,
        audio_embeddings,
        torch.rand((num_videos, num_events)),
        torch.rand((num_videos, num_events)),
    )
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    val_dataloader = DataLoader(dataset, batch_size=batch_size)

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

    optimizer = optim.Adam(model.parameters())
    criterion = LFAVLoss()

    model = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=2,
        device=device,
    )
