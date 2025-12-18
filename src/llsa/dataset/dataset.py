from enum import Enum
import json

import datasets
from datasets import load_dataset
import PIL.Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from typing import List, Optional
import os
import torch


class DataKey(Enum):
    IMAGE = 0
    INT_LABLE = 1
    TEXT = 2


def create_dataset(
    resolution,
    dataset_name=None,
    dataset_config_name=None,
    data_dir=None,
    data_files=None,
    dataset_key_map={DataKey.IMAGE: 'image'},
    split='train',
    center_crop=False,
    random_flip=False,
    downsample_mode='bilinear',
    cache_dir=None
):
    if dataset_name is not None:
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=cache_dir,
            split=split
        )
    elif data_files is not None:
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            split=split
        )
    elif data_dir is not None:
        dataset = load_dataset(
            "imagefolder",
            data_dir=data_dir,
            cache_dir=cache_dir,
            split=split)
    else:
        raise ValueError(
            '`dataset_name`, `data_files`, or `data_dir` must not be None')

    if downsample_mode == 'bilinear':
        resize_mode = transforms.InterpolationMode.BILINEAR
    elif downsample_mode == 'area' or downsample_mode == 'box':
        resize_mode = transforms.InterpolationMode.BOX
    else:
        raise NotImplementedError(
            f'downsample mode "{downsample_mode}" does not exist')

    augmentations = transforms.Compose(
        [
            transforms.Resize(
                resolution, interpolation=resize_mode),
            transforms.CenterCrop(
                resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        result_dict = {}

        if DataKey.IMAGE in dataset_key_map:
            dataset_image_key = dataset_key_map[DataKey.IMAGE]
            images = [augmentations(image.convert("RGB"))
                      for image in examples[dataset_image_key]]
            result_dict[DataKey.IMAGE] = images
        return result_dict

    dataset.set_transform(transform_images)

    return dataset


class DatasetWithMetadata(Dataset):
    def __init__(self, metafile_path: str):
        super().__init__()

        with open(metafile_path, 'rb') as f:
            items = json.load(f)['labels']

        self.items = items
        self.length = len(items)
        self.base_dir = os.path.dirname(metafile_path)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        file_name, label = self.items[index]
        return PIL.Image.open(os.path.join(self.base_dir, file_name)), int(label)


def create_imagefolder_dataloader(
    resolution: int,
    data_dir: Optional[str] = None,
    split: str = 'train',
    metadata_path: str = None,
    center_crop: bool = False,
    random_flip: bool = False,
    downsample_mode: str = 'bilinear',
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    shuffle: bool = True,
):

    assert data_dir is not None or metadata_path is not None

    if data_dir is not None:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise ValueError(
                f"{split_dir} not exists. Expect data_dir/{split}/<class>/...")

    if downsample_mode == 'bilinear':
        resize_mode = transforms.InterpolationMode.BILINEAR
    elif downsample_mode in ('area', 'box'):
        resize_mode = transforms.InterpolationMode.BOX
    else:
        raise NotImplementedError(
            f'downsample mode "{downsample_mode}" is not supported')

    augmentations = transforms.Compose([
        transforms.Resize(resolution, interpolation=resize_mode),
        transforms.CenterCrop(
            resolution) if center_crop else transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    if metadata_path is None:
        dataset = datasets.ImageFolder(split_dir, transform=None)
    else:
        dataset = DatasetWithMetadata(metadata_path)

    def collate_fn(batch: List):
        imgs, labels = zip(*batch)

        processed_imgs = [augmentations(img.convert("RGB"))
                          for img in imgs]

        images_tensor = torch.stack(processed_imgs, dim=0)      # [B, 3, H, W]
        labels_tensor = torch.as_tensor(labels, dtype=torch.long)

        return {
            DataKey.IMAGE: images_tensor,
            DataKey.INT_LABLE: labels_tensor,
        }

    shuffle = shuffle

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    return dataset, dataloader
