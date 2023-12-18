import csv
from glob import glob

import torch
import numpy as np
from torch.utils.data import TensorDataset

DATA_PATH = "/work/vig/Datasets/LFAV/features/{model}"
ADJUSTED_SNIPPET_COUNT = 200
IGNORE_LABELS = ["toilet_flush", "train", "silent"]
NUM_EVENTS = 35


def _load_data_model(model):
    data = {}
    for file in glob(DATA_PATH.format(model=model) + "/*.npy"):
        data[file.split("/")[-1].split(".")[0]] = np.load(file)
    return data


def _adjust_snippet_count(data, original_count, snippet_count=ADJUSTED_SNIPPET_COUNT):
    if original_count > snippet_count:
        interval = original_count // snippet_count
        new_data = data[::interval, :]
        return new_data[:snippet_count]  # in case not perfect multiple
    elif original_count < snippet_count:
        repeat_n = snippet_count // original_count
        remaining = snippet_count % original_count
        new_data = np.concatenate((np.repeat(data, repeat_n, axis=0), data[:remaining]))
        return new_data
    else:
        return data


def map_snippet_number(
    original_count, original_snippet_number, snippet_count=ADJUSTED_SNIPPET_COUNT
):
    if original_count > snippet_count:
        interval = original_count // snippet_count
        return original_snippet_number // interval
    elif original_count < snippet_count:
        repeat_n = snippet_count // original_count
        return original_snippet_number * repeat_n
    else:
        return original_snippet_number


def _create_label_index():
    """Label integer index for one-hot encoding."""
    labels = [
        "guitar",
        "drum",
        "violin",
        "piano",
        "accordion",
        "banjo",
        "cello",
        "shofar",
        "speech",
        "singing",
        "cry",
        ("laughing", "laughter"),
        "clapping",
        "cheering",
        "dance",
        "dog",
        "cat",
        "chicken_rooster",
        "horse",
        "rodents",
        "car",
        "helicopter",
        "fixed-wing_aircraft",
        "bicycle",
        "alarm",
        "chainsaw",
        "car_alarm",
        "frisbee",
        "playing_basketball",
        "playing_baseball",
        "playing_badminton",
        "playing_volleyball",
        "playing_tennis",
        "playing_ping-pong",
        "playing_soccer",
    ]
    label_index = {}
    for i, label in enumerate(labels):
        if isinstance(label, tuple):
            for label_part in label:
                label_index[label_part] = i
        else:
            label_index[label] = i
    return label_index


def _load_vl_labels_one(modality, split, label_index):
    """Load video-level labels as one-hot vectors."""
    assert modality in ["visual", "audio"]
    assert split in ["train", "val", "test"]
    path = f"/work/vig/Datasets/LFAV/annotations/{split}/{split}_{modality}_weakly.csv"
    labels = []
    for vals in list(csv.reader(open(path, "r"), delimiter="\t"))[1:]:
        label = torch.zeros(NUM_EVENTS)
        for label_name in vals[1].split(","):
            if label_name.strip() not in IGNORE_LABELS:
                label[label_index[label_name.strip()]] = 1
        labels.append(label)
    return torch.stack(labels)


def _load_vl_labels(split):
    label_index = _create_label_index()
    if split == "train":
        visual = _load_vl_labels_one("visual", "train", label_index)
        audio = _load_vl_labels_one("audio", "train", label_index)
    elif split == "val":
        visual = _load_vl_labels_one("visual", "val", label_index)
        audio = _load_vl_labels_one("audio", "val", label_index)
    elif split == "test":
        visual = _load_vl_labels_one("visual", "test", label_index)
        audio = _load_vl_labels_one("audio", "test", label_index)
    else:
        raise Exception(f"Split {split} should be in ['train', 'val', 'test'].")

    return visual, audio


def _load_data(ids, split, device, overfit=None):
    r2plus1d = _load_data_model("r2plus1d")
    resnet18 = _load_data_model("resnet18")
    vggish = _load_data_model("vggish")

    for model in [r2plus1d, resnet18, vggish]:
        for key in model.keys():
            original_snippet_count = model[key].shape[0]
            model[key] = _adjust_snippet_count(model[key], original_snippet_count)

    r2plus1d = {k: v for k, v in r2plus1d.items() if k in ids}
    resnet18 = {k: v for k, v in resnet18.items() if k in ids}
    vggish = {k: v for k, v in vggish.items() if k in ids}

    r2plus1d = torch.stack(tuple(torch.from_numpy(x) for x in r2plus1d.values()))
    resnet18 = torch.stack(tuple(torch.from_numpy(x) for x in resnet18.values()))
    vggish = torch.stack(tuple(torch.from_numpy(x) for x in vggish.values()))

    video_embeddings = torch.cat((r2plus1d, resnet18), dim=2).to(device)
    audio_embeddings = vggish.to(device)

    visual_labels, audio_labels = _load_vl_labels(split)
    assert len(video_embeddings) == len(visual_labels)
    assert len(audio_embeddings) == len(audio_labels)

    if overfit is not None:  # already seeded
        indices = torch.randperm(len(video_embeddings))[:overfit]
        video_embeddings = video_embeddings[indices]
        audio_embeddings = audio_embeddings[indices]
        visual_labels = visual_labels[indices]
        audio_labels = audio_labels[indices]

    dataset = TensorDataset(
        video_embeddings, audio_embeddings, visual_labels, audio_labels
    )
    return dataset


def load_train_data(device, overfit_batch=False, batch_size=None):
    if overfit_batch:
        assert batch_size is not None
    path = "/work/vig/Datasets/LFAV/annotations/train/train_audio_weakly.csv"
    ids = [vals[0] for vals in list(csv.reader(open(path, "r"), delimiter="\t"))[1:]]
    data = _load_data(ids, "train", device, overfit=batch_size)
    return data


def load_val_data(device):
    path = "/work/vig/Datasets/LFAV/annotations/val/val_audio_weakly.csv"
    ids = [vals[0] for vals in list(csv.reader(open(path, "r"), delimiter="\t"))[1:]]
    data = _load_data(ids, "val", device)
    return data


def load_test_data(device):
    path = "/work/vig/Datasets/LFAV/annotations/test/test_audio_weakly.csv"
    ids = [vals[0] for vals in list(csv.reader(open(path, "r"), delimiter="\t"))[1:]]
    data = _load_data(ids, "test", device)
    return data


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        len(load_train_data(device))
        + len(load_val_data(device))
        + len(load_test_data(device))
    )
