import torch

from logging import getLogger

_GLOBAL_SEED = 0
logger = getLogger()


def make_image_dataset(
    data_paths,
    batch_size,
    transform=None,
    num_workers=10,
    pin_mem=True,
    shuffle=False
):
    dataset = ImageDataset(
        data_paths=data_paths,
        transform=transform)

    logger.info('ImageDataset dataset created')
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        shuffle=shuffle)

    return dataset, data_loader


class ImageDataset(torch.utils.data.Dataset):
    """ Image immunization dataset. """

    def __init__(
        self,
        data_paths,
        transform=None,
    ):
        self.data_paths = data_paths
        self.transform = transform   

        # Load image paths and labels
        samples, masks, prompts = [], []

        self.samples = samples
        self.masks = masks
        self.prompts = prompts


    def __getitem__(self, index):
        sample = self.samples[index]
        mask = self.mask[index]
        prompt = self.prompt[index]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, mask, prompt


    def __len__(self):
        return len(self.samples)
