from torch.utils.data import Dataset


class LFAVDataset(Dataset):
    """Dataset class for the LFAV dataset."""

    def __init__(self, video_embeddings, audio_embeddings, labels):
        """Initialize the LFAVDataset.

        Args:
            video_embeddings: Video embeddings of (num_videos, num_snippets, video_dim)
            audio_embeddings: Audio embeddings of (num_videos, num_snippets, audio_dim)
            labels: Labels of (num_videos, num_classes)
        """
        self.video_embeddings = video_embeddings
        self.audio_embeddings = audio_embeddings
        self.labels = labels

    def __getitem__(self, idx):
        return (
            self.video_embeddings[idx],
            self.audio_embeddings[idx],
            self.labels[idx],
        )

    def __len__(self):
        return len(self.video_embeddings)