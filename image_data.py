from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset


class config:
    dataset_id = ""
    batch_size = 32
    split = 10000


hfdata = load_dataset(config.dataset_id, split="train", streaming=True)
hfdata = hfdata.take(config.split)


class ImageNet(IterableDataset):
    def __init__(self, dataset=hfdata):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            image = read_image(item["image"])
            label = item["label"]

            image = torch.tensor(image, dtype=config.dtype)
            label = torch.tensor(label, dtype=config.dtype)

            yield image, label


img_dataset = ImageNet()
train_loader = DataLoader(img_dataset, config.batch_size, shuffle=True)
