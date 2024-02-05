# This file contains the code for synthetic cloud microphysics dataset loaders for VIP-CT.
# It is based on PyTorch3D source code ('https://github.com/facebookresearch/pytorch3d') by FAIR
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite the paper described in the readme file:
# Roi Ronen, Vadim Holodovsky and Yoav. Y. Schechner, "Variable Imaging Projection Cloud Scattering Tomography",
# Proc. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.
#
# Copyright (c) Roi Ronen. The python code is available for
# non-commercial use and exploration.  For commercial use contact the
# author. The author is not liable for any damages or loss that might be
# caused by use or connection to this code.
# All rights reserved.
#
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.


import os, glob
from typing import Tuple
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import socket
import random
import scipy.io as sio


DEFAULT_DATA_ROOT = '/wdata/inbalkom/NN_Data'

ALL_DATASETS = ("BOMEX_10cams_20m_polarization", "CASS_10cams_20m_polarization", "BOMEX_50CCN", "BOMEX_500CCN_10cams_20m_polarization_at3d")


def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    batch = np.array(batch, dtype=object).transpose().tolist()
    return batch


def get_cloud_microphysics_datasets(
    cfg,
    data_root: str = DEFAULT_DATA_ROOT,
) -> Tuple[Dataset, Dataset]:
    """
    Obtains the training and validation dataset object for a dataset specified
    with the `dataset_name` argument.

    Args:
        dataset_name: The name of the dataset to load.
        image_size: A tuple (height, width) denoting the sizes of the loaded dataset images.
        data_root: The root folder at which the data is stored.

    Returns:
        train_dataset: The training dataset object.
        val_dataset: The validation dataset object.
        test_dataset: The testing dataset object.
    """
    dataset_name = cfg.data.dataset_name

    if dataset_name not in ALL_DATASETS:
        raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")

    if dataset_name == 'BOMEX_10cams_20m_polarization':
        data_root = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/'
        data_root_gt_train = os.path.join(data_root, "train")
        data_root_gt_val = os.path.join(data_root, "test")
        # image_size = [123, 123]
        cfg.data.image_size = [116, 116]
    elif dataset_name == 'BOMEX_500CCN_10cams_20m_polarization_at3d':
        data_root = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_AT3D/const_env_params/'
        data_root_gt_train = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/train'
        data_root_gt_val = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/train'
        # image_size = [123, 123]
        cfg.data.image_size = [116, 116]
    elif dataset_name == "CASS_10cams_20m_polarization":
        data_root = '/wdata/inbalkom/NN_Data/CASS_50m_256x256x139_600CCN/64_64_32_cloud_fields/'
        data_root_gt_train = os.path.join(data_root, "train")
        data_root_gt_val = os.path.join(data_root, "test")
        cfg.data.image_size = [315, 315]
    elif dataset_name == "BOMEX_50CCN":
        data_root = "/wdata/inbalkom/NN_Data/BOMEX_32x32x64_50CCN_50m/"
        data_root_gt_train = os.path.join(data_root, "train")
        data_root_gt_val = os.path.join(data_root, "test")
        cfg.data.image_size = [116, 116]

    print(f"Loading dataset {dataset_name}, image size={str(cfg.data.image_size)} ...")
    data_train_paths = [f for f in glob.glob(os.path.join(data_root, "train/cloud*.pkl"))]

    train_len = cfg.data.n_training if cfg.data.n_training>0 else len(data_train_paths)
    data_train_paths = data_train_paths[:train_len]

    n_cam = cfg.data.n_cam
    mean = cfg.data.mean
    std = cfg.data.std
    rand_cam = cfg.data.rand_cam
    train_dataset = MicrophysicsCloudDataset(
            data_train_paths,
        data_root_gt_train,
        n_cam=n_cam,
        rand_cam = rand_cam,
        mask_type=cfg.ct_net.mask_type,
        mean=mean,
        std=std,
        dataset_name = dataset_name,
    )

    if dataset_name == 'BOMEX_500CCN_10cams_20m_polarization_at3d':
        val_paths = [f for f in glob.glob(os.path.join(data_root, "validation/cloud*.pkl"))]
    else:
        val_paths = [f for f in glob.glob(os.path.join(data_root, "test/cloud*.pkl"))]

    val_len = cfg.data.n_val if cfg.data.n_val > 0 else len(val_paths)
    val_paths = val_paths[:val_len]
    val_dataset = MicrophysicsCloudDataset(val_paths, data_root_gt_val, n_cam=n_cam,
        rand_cam = rand_cam, mask_type=cfg.ct_net.val_mask_type, mean=mean, std=std,   dataset_name = dataset_name)
    return train_dataset, val_dataset


class MicrophysicsCloudDataset(Dataset):
    def __init__(self, cloud_dir, gt_grids_dir, n_cam, rand_cam=False, transform=None, target_transform=None, mask_type=None, mean=0, std=1, dataset_name=''):
        self.cloud_dir = cloud_dir
        self.gt_grids_dir = gt_grids_dir
        self.transform = transform
        self.target_transform = target_transform
        self.mask_type = mask_type
        self.n_cam = n_cam
        self.rand_cam = rand_cam
        self.mean = mean
        self.std = std
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.cloud_dir)

    def __getitem__(self, idx):
        cloud_path = self.cloud_dir[idx]
        # if 'TEST' in cloud_path:
        #     data_root = os.path.join(DEFAULT_DATA_ROOT, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras', 'test')
        # else:
        #     data_root = os.path.join(DEFAULT_DATA_ROOT, 'BOMEX_256x256x100_5000CCN_50m_micro_256', '10cameras', 'train')
        # image_index = cloud_path.split('satellites_images_')[-1].split('.pkl')[0]
        # projection_path = os.path.join(data_root, f"cloud_results_{image_index}.pkl")
        image_index = cloud_path.split('cloud_results_')[-1].split('.pkl')[0]
        gt_grid_path = os.path.join(self.gt_grids_dir, "cloud_results_"+str(image_index)+".pkl")
        projection_path = cloud_path
        with open(cloud_path, 'rb') as f:
            data = pickle.load(f)
        with open(gt_grid_path, 'rb') as f:
            gt_grid_data = pickle.load(f)
        images = data['images'] #np.concatenate((data['images'], data['DoLPs'][:, None], data['AoLPs'][:, None]), 1)
        mask = None
        if self.mask_type == 'space_carving':
            if self.dataset_name == "CASS_10cams_20m_polarization":
                cloud_path2 = os.path.join('/wdata/roironen/Data/CASS_50m_256x256x139_600CCN/10cameras_20m', cloud_path.split("/")[-2], cloud_path.split("/")[-1])
                with open(cloud_path2, 'rb') as f:
                    data2 = pickle.load(f)
                mask = data2['mask']
            else:
                mask = data['mask']

        elif self.mask_type == 'space_carving_morph':
            mask = data['mask_morph']
        elif self.mask_type == 'gt_mask':
            mask = (data['lwc_gt'] > 0) * (data['reff_gt'] > 0) * (data['veff_gt'] > 0)
            if mask.shape[0] == 74 and mask.shape[1] == 74 and mask.shape[2] == 37:
                mask = mask[5:-5, 5:-5, :-5]
        if mask.dtype!='bool':
            mask = mask>0
        if 'varying' in self.dataset_name:
            index = torch.randperm(10)[0]
            cam_i = torch.arange(index,self.n_cam*10,self.n_cam)
            mask = mask[index] if mask is not None else None
        else:
            cam_i = torch.arange(self.n_cam)
        if self.dataset_name == "CASS_10cams_20m_polarization":
            images = np.squeeze(images[:,cam_i,:,:,:])
        else:
            images = images[cam_i, :, :, :]
        images -= np.array(self.mean).reshape((1,3,1,1))
        images /= np.array(self.std).reshape((1,3,1,1))

        microphysics = np.array([gt_grid_data['lwc_gt'],gt_grid_data['reff_gt'],gt_grid_data['veff_gt']])

        if "CASS" in self.dataset_name:
            assert (microphysics.shape[1] == 74 and microphysics.shape[2] == 74 and microphysics.shape[3] == 37) or (
                        microphysics.shape[1] == 64 and microphysics.shape[2] == 64 and microphysics.shape[3] == 32)
            if microphysics.shape[1] == 74 and microphysics.shape[2] == 74 and microphysics.shape[3] == 37:
                microphysics = microphysics[:,5:-5, 5:-5, :-5]
        elif "BOMEX" in self.dataset_name:
            assert (microphysics.shape[1] == 32 and microphysics.shape[2] == 32 and (microphysics.shape[3] == 32 or microphysics.shape[3] == 64))

        if hasattr(data, 'image_sizes'):
            image_sizes = data['image_sizes'][cam_i]
        else:
            image_sizes = [image.shape[1:] for image in images]

        grid = data['grid']
        if grid.dtype=='float64':
            grid = np.float32(grid)
        if "CASS" in self.dataset_name:
            if grid[0].shape[0] == 74 and grid[1].shape[0] == 74 and grid[2].shape[0] == 37:
                grid[0] = grid[0][5:-5]
                grid[1] = grid[1][5:-5]
                grid[2] = grid[2][:-5]
        camera_center = data['cameras_pos'][cam_i]
        projection_matrix = data['cameras_P'][cam_i]
        # with open('/wdata/inbalkom/NN_Data/tmp/projection_matrices.pkl', 'rb') as f:
        #     projection_matrix = pickle.load(f)[cam_i]
        return images, microphysics, grid, image_sizes, projection_matrix, camera_center, mask
