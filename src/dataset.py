import csv
from glob import glob

import torch
import numpy as np
from torch.utils.data import TensorDataset

DATA_PATH = "/work/vig/Datasets/LFAV/features/{model}"
MODELS = ["r2plus1d", "resnet18", "vggish"]
ADJUSTED_SNIPPET_COUNT = 200


# TODO double check if joint data is useful
# TODO get the labels
# TODO convert to one-hot and pass in


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


def _load_data(ids, device):
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

    dataset = TensorDataset(video_embeddings, audio_embeddings)
    return dataset


def load_train_data(device):
    path = "/work/vig/Datasets/LFAV/annotations/train/train_audio_weakly.csv"
    ids = [vals[0] for vals in list(csv.reader(open(path, "r"), delimiter="\t"))[1:]]
    data = _load_data(ids, device)
    return data


def load_val_data(device):
    path = "/work/vig/Datasets/LFAV/annotations/val/val_audio_weakly.csv"
    ids = [vals[0] for vals in list(csv.reader(open(path, "r"), delimiter="\t"))[1:]]
    data = _load_data(ids, device)
    return data


def load_test_data(device):
    path = "/work/vig/Datasets/LFAV/annotations/test/test_audio_weakly.csv"
    ids = [vals[0] for vals in list(csv.reader(open(path, "r"), delimiter="\t"))[1:]]
    data = _load_data(ids, device)
    return data


if __name__ == "__main__":
    # dataset = load_data()
    # print(len(dataset))
    # print(dataset[0][0].shape)
    # print(dataset[0][1].shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        len(load_train_data(device))
        + len(load_val_data(device))
        + len(load_test_data(device))
    )
