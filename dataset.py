from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, file_list, lables, transform=None):
        self.file_list = file_list
        self.labels = lables
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_list[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)

