from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from os.path import join


class TSPCV_Dataset(Dataset):
    def __init__(self, path, transform=None, data_dir="data"):

        self.data = pd.read_csv(path, sep=",")
        self.transform = transform
        self.data_dir = data_dir

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        img = Image.open(join(self.data_dir, self.data["filename"][index]))

        if self.transform:
            img = self.transform(img)

        try:
            distance = float(self.data["distance"][index])
            return img, distance
        except KeyError:
            return img
