import torch
import torch.utils.data as udata


class ImageNetDataset(udata.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]
        if self.transform is not None:
            X = self.transform(X)

        return X, y