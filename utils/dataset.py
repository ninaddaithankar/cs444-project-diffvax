# implementation for the 'clothing co-parsing' dataset: https://github.com/bearpaw/clothing-co-parsing/tree/master 

import os
import torch

from PIL import Image
from scipy.io import loadmat

from logging import getLogger



_GLOBAL_SEED = 0
logger = getLogger()



def get_train_and_val_loaders(
    dataset,
    train_split_pct,
    batch_size,
    num_workers=4,
    pin_mem=True,
    train_shuffle=True,
    val_shuffle=False,
):
    # define sizes for train and validation
    train_size = int(train_split_pct * len(dataset))
    val_size = len(dataset) - train_size

    # randomly split the dataset
    train_dataset, val_dataset = torch.utils.random_split(dataset, [train_size, val_size])

    print("Training dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))

    # create the dataloaders
    train_loader = torch.utils.DataLoader(
        train_datasetbatch_size=batch_size,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        shuffle=train_shuffle)
    
    val_loader = torch.utils.DataLoader(val_datasetbatch_size=batch_size,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        shuffle=val_shuffle)

    return train_dataset, train_loader, val_dataset, val_loader



class CC2_Dataset(torch.utils.data.Dataset):
    """ Image immunization dataset. """

    def __init__(
        self,
        dataset_path,
        image_transforms=None,
        mask_transforms=None,
        shared_transforms=None,
        img_dir="/photos/",
        masks_dir="/annotations/pixel-level/"
    ):
        self.dataset_paths = dataset_path
        self.images_dir = os.path.join(dataset_path, img_dir)
        self.masks_dir = os.path.join(dataset_path, masks_dir)

        self.image_transforms = image_transforms   
        self.mask_transforms = mask_transforms
        self.shared_transforms = shared_transforms   

        # load images, masks, and prompt paths
        self.images = sorted(os.listdir(self.images_dir))
        self.masks = sorted(os.listdir(self.masks_dir))

        assert(len(self.images) == len(self.masks))

        # TODO: implement prompt loading
        prompts = []
        self.prompts = prompts


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        image = Image.open(os.path.join(self.images_dir, self.images[index])).convert("RGB")
        
        # the mask here is an annotated image where the background is 0 and other pixels are annotated with class numbers which we don't need
        mask = loadmat(os.path.join(self.masks_dir, self.masks[index]))

        # so we simply keep the background as 0 and set other areas to 1 and treat the 1 are as foreground
        mask[mask != 0] = 1

        # TODO: change this placeholder to actual prompts
        prompt = "a person in a courtroom"

        if self.shared_transforms is not None:
            image = self.shared_transforms(image)
            mask = self.shared_transforms(mask)

        if self.image_transforms is not None:
            image = self.image_transforms(image)

        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)

        return image, mask, prompt


