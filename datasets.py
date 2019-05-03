from os.path import join, exists, isdir

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import CenterCrop, Compose, Normalize, ToTensor

from mask_generators import ImageMaskGenerator


def compute_normalization(data, one_hot_max_sizes):
    """
    Compute the normalization parameters (i. e. mean to subtract and std
    to divide by) for each feature of the dataset.
    For categorical features mean is zero and std is one.
    i-th feature is denoted to be categorical if one_hot_max_sizes[i] >= 2.
    Returns two vectors: means and stds.
    """
    norm_vector_mean = torch.zeros(len(one_hot_max_sizes))
    norm_vector_std = torch.ones(len(one_hot_max_sizes))
    for i, size in enumerate(one_hot_max_sizes):
        if size >= 2:
            continue
        v = data[:, i]
        v = v[1 - torch.isnan(v)]
        vmin, vmax = v.min(), v.max()
        vmean = v.mean()
        vstd = v.std()
        norm_vector_mean[i] = vmean
        norm_vector_std[i] = vstd
    return norm_vector_mean, norm_vector_std


class CelebA(Dataset):
    """CelebA dataset."""

    def __init__(self, root_dir, partition_file, mode, transform=None):
        """
        Args:
            root_dir (string):       Directory with all the images.
            partition_file (string): File with the partition list.
            mode (string):           Used part of dataset:
                                     train, test or valid.
            transform (callable,
                       optional):    Optional transform to be applied
                                     on a sample.
        """
        if not exists(root_dir):
            err = 'Celeba aligned images directory is not found: %s' % root_dir
            raise FileNotFoundError(err)
        if not isdir(root_dir):
            err = '%s must be a directory with aligned images' % root_dir
            raise NotADirectoryError(err)
        if not exists(partition_file):
            err = 'Celeba partition file is not found: %s' % partition_file
            raise FileNotFoundError(err)

        self.root_dir = root_dir
        self.partition = {
            'train': [],
            'test': [],
            'valid': []
        }
        part = {
            '0': 'train',
            '1': 'valid',
            '2': 'test'
        }
        for line in open(partition_file):
            if not line.strip():
                continue
            filename, part_id = line.strip().split(' ')
            self.partition[part[part_id]].append(filename)
        if mode not in self.partition.keys():
            err = "Mode must be 'train', 'valid' or 'test', "
            err += "but %s got instead."
            err = err % str(mode)
            raise ValueError(err)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.partition[self.mode])

    def __getitem__(self, idx):
        img_name = join(self.root_dir,
                        self.partition[self.mode][idx])
        image = default_loader(img_name)

        if self.transform is not None:
            image = self.transform(image)

        return image


class LengthBounder(Dataset):
    """Dataset wrapper which bounds the length of the underlying dataset."""
    def __init__(self, dataset, max_length):
        self.dataset = dataset
        self.max_length = max_length

    def __len__(self):
        return min(len(self.dataset), self.max_length)

    def __getitem__(self, idx):
        return self.dataset[idx]


class ZipDatasets(Dataset):
    """
    Dataset wrapper which returns a list of objects
    from a number of datasets.
    It behaves like standard zip(dataset_1, dataset_2, ...),
    i. e. ZipDataset(dataset_1, dataset_2, ...)[i] is
    [dataset_1[i], dataset_2[i], ...]
    """
    def __init__(self, *args):
        self.args = args

    def __len__(self):
        return min(len(arg) for arg in self.args)

    def __getitem__(self, idx):
        return [arg[idx] for arg in self.args]


class GeneratorDataset(Dataset):
    """
    Generates dataset by applying generator to each object
    of the original dataset.
    Used to generate masks for inpainting on the test set.
    """
    def __init__(self, generator, original_dataset, batch_size=16):
        self.generator = generator
        self.batch_size = batch_size
        self.original_dataset = original_dataset

        self.size = len(original_dataset)
        self.data = []
        idx = 0
        for idx in range(0, self.size, self.batch_size):
            cond_batch = []
            for j in range(self.batch_size):
                cond = original_dataset[min(j + idx,
                                            len(original_dataset) - 1)]
                cond_batch.append(cond[None])
            cond_batch = torch.cat(cond_batch)
            batch = generator(cond_batch)
            self.data.append(batch)
        self.data = torch.cat(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def load_dataset(name):
    """
    Returns dataset for image inpainting.
    Now returns only CelebA dataset (train, validation and test parts of it)
    and generated masks for the test part.
    """
    celeba_transforms = Compose([
        CenterCrop(128),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    celeba_root_dir = '/dbstore/datasets/celebA'  # change it for your system!
    celeba_img_dir = join(celeba_root_dir, 'img_align_celeba')
    celeba_partition = join(celeba_root_dir, 'list_eval_partition.txt')

    if name == 'celeba_train':
        return CelebA(
            celeba_img_dir,
            celeba_partition,
            'train',
            celeba_transforms
        )
    elif name == 'celeba_val':
        # in order to speed up training we restrict validation set
        # to have only 1024 images
        return LengthBounder(CelebA(
            celeba_img_dir,
            celeba_partition,
            'valid',
            celeba_transforms), 1024)
    elif name == 'celeba_test':
        # in order to demonstrate the inpainting results we don't need
        # the whole test set, so we use 256 test images only
        return LengthBounder(CelebA(
            celeba_img_dir,
            celeba_partition,
            'test',
            celeba_transforms), 256)
    elif name == 'celeba_inpainting_masks':
        return GeneratorDataset(ImageMaskGenerator(),
                                load_dataset('celeba_test'))
    else:
        raise ValueError('Unknown dataset %s' % str(name))
