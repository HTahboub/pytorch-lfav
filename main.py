from glob import glob

import numpy as np

DATA_PATH = "/work/vig/Datasets/LFAV/features/{model}"
MODELS = ["r2plus1d", "resnet18", "vggish"]

def load_data(model):
    data = {}
    for file in glob(DATA_PATH.format(model=model) + "/*.npy"):
        data[file.split("/")[-1].split(".")[0]] = np.load(file)
    return data


if __name__ == "__main__":
    for model in MODELS:
        data = load_data(model)
        print(model, len(data))
        for key, value in list(data.items())[:5]:
            print(key, value.shape)
        print()
