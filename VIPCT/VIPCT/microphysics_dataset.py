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

ALL_DATASETS = ("BOMEX_pretty_clouds","BOMEX_polarization_pyshdom_varsun_scatIDA","BOMEX_polarization_pyshdom_varsun_scatIQU","BOMEX_polarization_pyshdom_varsun", "BOMEX_polarization_pyshdom_varying_M", "CASS_BOMEX_polarization_pyshdom","CASS_10cams_20m_polarization_pyshdom","BOMEX_500CCN_10cams_20m_polarization_pyshdom",
                "BOMEX_500CCN_10cams_20m_polarization_at3d_const","BOMEX_500CCN_10cams_20m_polarization_at3d_united", "BOMEX_500CCN_10cams_20m_polarization_at3d_varwind",
                "BOMEX_500CCN_10cams_20m_polarization_at3d_varsun","BOMEX_500CCN_10cams_20m_polarization_at3d_varsat_varwind", "BOMEX_500CCN_10cams_20m_polarization_at3d_varsat_varenv"
                "CASS_10cams_20m_polarization_at3d_const", "BOMEX_50CCN_pyshdom", "LES_pyshdom", "BOMEX_pyshdom_varsun_roi")


def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    batch = np.array(batch, dtype=object).transpose().tolist()
    return batch


def get_cloud_microphysics_datasets(cfg):
    """
    Obtains the training and validation dataset object for a dataset specified
    with the `dataset_name` argument.

    Args:
        cfg: Set of parameters for the datasets.

    Returns:
        train_dataset: The training dataset object.
        val_dataset: The validation dataset object.
        test_dataset: The testing dataset object.
    """
    dataset_name = cfg.data.dataset_name

    if dataset_name not in ALL_DATASETS:
        raise ValueError(f"'{dataset_name}' does not refer to a known dataset.")

    if 'BOMEX_500CCN_10cams_20m_polarization_at3d' in dataset_name:
        data_root = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_AT3D/'
        data_root_gt_train = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/train'
        data_root_gt_val = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/train'
        data_root_gt_test = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/test'
        cfg.data.image_size = [116, 116]
    elif 'CASS_10cams_20m_polarization_at3d' in dataset_name:
        data_root = '/wdata_visl/inbalkom/NN_Data/CASS_50m_256x256x139_600CCN/64_64_32_cloud_fields/CloudCT_SIMULATIONS_AT3D/'
        data_root_gt_train = '/wdata/inbalkom/NN_Data/CASS_50m_256x256x139_600CCN/64_64_32_cloud_fields/train'
        data_root_gt_val = '/wdata/inbalkom/NN_Data/CASS_50m_256x256x139_600CCN/64_64_32_cloud_fields/train'
        data_root_gt_test = '/wdata/inbalkom/NN_Data/CASS_50m_256x256x139_600CCN/64_64_32_cloud_fields/test'
        cfg.data.image_size = [315, 315]
    elif 'BOMEX_500CCN_10cams_20m_polarization_pyshdom' in dataset_name:
        data_root = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/'
        data_root_gt_train = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/train'
        data_root_gt_val = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/validation'
        data_root_gt_test = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/clouds/test'
    elif "CASS_10cams_20m_polarization_pyshdom" in dataset_name:
        data_root = '/wdata/inbalkom/NN_Data/CASS_50m_256x256x139_600CCN/64_64_32_cloud_fields/'
        data_root_gt_train = '/wdata/inbalkom/NN_Data/CASS_50m_256x256x139_600CCN/64_64_32_cloud_fields/train'
        data_root_gt_val = '/wdata/inbalkom/NN_Data/CASS_50m_256x256x139_600CCN/64_64_32_cloud_fields/validation'
        data_root_gt_test = '/wdata/inbalkom/NN_Data/CASS_50m_256x256x139_600CCN/64_64_32_cloud_fields/test'
    elif "CASS_BOMEX_polarization_pyshdom" in dataset_name:
        data_root_cass = '/wdata/inbalkom/NN_Data/CASS_50m_256x256x139_600CCN/64_64_32_cloud_fields/'
        data_root_bomex = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/'
        data_root_gt_train = None
        data_root_gt_val = None
        data_root_gt_test = None
    elif "BOMEX_polarization_pyshdom_varying_M" in dataset_name:
        data_root = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/var_sats/'
        data_root_gt_train = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/train'
        data_root_gt_val = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/validation'
        data_root_gt_test = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/test'
        cfg.data.image_size = [116, 116]
    elif "BOMEX_pyshdom_varsun_roi" in dataset_name:
        data_root = '/wdata/roironen/Data/BOMEX_128x128x100_5000CCN_50m_micro_256/10cameras_20m_varying_sun/train/'
        data_root_gt_train = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/train'
        data_root_gt_val = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/test'
        data_root_gt_test = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/test'
        cfg.data.image_size = [116, 116]
    elif "BOMEX_polarization_pyshdom_varsun" in dataset_name:
        data_root = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/var_sun/'
        data_root_gt_train = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/train'
        data_root_gt_val = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/validation'
        data_root_gt_test = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/test'
        cfg.data.image_size = [116, 116]
    elif dataset_name == "BOMEX_50CCN_pyshdom":
        data_root = "/wdata/inbalkom/NN_Data/BOMEX_32x32x64_50CCN_50m/"
        data_root_gt_train = os.path.join(data_root, "train")
        data_root_gt_val = os.path.join(data_root, "test")
        cfg.data.image_size = [116, 116]
    elif dataset_name == "LES_pyshdom":
        data_root = '/wdata_visl/inbalkom/NN_Data/LES_clouds_for_paper/'
        data_root_gt_train = None
        data_root_gt_val = None
        data_root_gt_test = '/wdata_visl/inbalkom/NN_Data/LES_clouds_for_paper/test'
    elif dataset_name == "BOMEX_pretty_clouds":
        data_root = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/pretty_clouds/'
        data_root_gt_train = None
        data_root_gt_val = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/validation'
        data_root_gt_test = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/pretty_clouds/test'


    if 'at3d' in dataset_name:
        if 'varsat' in dataset_name:
            data_root = os.path.join(data_root, "varying_sats_loc")
        if 'const' in dataset_name:
            data_root = os.path.join(data_root, "const_env_params")
        elif 'varwind' in dataset_name:
            data_root = os.path.join(data_root, "varying_wind_const_sun")
        elif 'varsun' in dataset_name:
            data_root = os.path.join(data_root, "varying_sun_const_wind")
        elif 'varenv' in dataset_name:
            data_root = os.path.join(data_root, "varying_env_params")
        elif 'united' in dataset_name:
            data_root = os.path.join(data_root, "united_const_varwind")

    if dataset_name == "BOMEX_pyshdom_varsun_roi":
        print(f"Loading dataset {dataset_name}, image size={str(cfg.data.image_size)} ...")
        data_train_paths = [f for f in glob.glob(os.path.join(data_root, "cloud*.pkl"))]
        image_indices = [int(cloud_path.split('cloud_results_')[-1].split('.pkl')[0]) for cloud_path in data_train_paths]
        data_train_paths = [data_train_path for ind, data_train_path in
                            enumerate(data_train_paths) if np.array(image_indices[ind]) <= 6000]
    elif dataset_name == "BOMEX_pretty_clouds":
        data_root_train = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/train'
        data_train_paths = [f for f in glob.glob(os.path.join(data_root_train, "cloud*.pkl"))]
    elif not dataset_name == 'CASS_BOMEX_polarization_pyshdom':
        print(f"Loading dataset {dataset_name}, image size={str(cfg.data.image_size)} ...")
        data_train_paths = [f for f in glob.glob(os.path.join(data_root, "train/cloud*.pkl"))]
    else:
        print(f"Loading dataset {dataset_name}...")
        data_train_paths_cass = [f for f in glob.glob(os.path.join(data_root_cass, "train/cloud*.pkl"))]
        data_train_paths_bomex = [f for f in glob.glob(os.path.join(data_root_bomex, "train/cloud*.pkl"))]
        maxind = min(len(data_train_paths_cass), len(data_train_paths_bomex))
        data_train_paths = data_train_paths_cass[:maxind] + data_train_paths_bomex[:maxind]
        random.shuffle(data_train_paths)

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

    if dataset_name == "BOMEX_pyshdom_varsun_roi":
        val_paths = [f for f in glob.glob(os.path.join(data_root, "cloud*.pkl"))]
        image_indices = [int(cloud_path.split('cloud_results_')[-1].split('.pkl')[0]) for cloud_path in val_paths]
        val_paths = [val_path for ind, val_path in
                            enumerate(val_paths) if np.array(image_indices[ind]) > 6000]
    elif dataset_name == "BOMEX_pretty_clouds":
        data_root_val = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/validation'
        val_paths = [f for f in glob.glob(os.path.join(data_root_val, "cloud*.pkl"))]
    elif not dataset_name == 'CASS_BOMEX_polarization_pyshdom':
        val_paths = [f for f in glob.glob(os.path.join(data_root, "validation/cloud*.pkl"))]
    else:
        val_paths = [f for f in glob.glob(os.path.join(data_root_cass, "validation/cloud*.pkl"))]
        val_paths += [f for f in glob.glob(os.path.join(data_root_bomex, "validation/cloud*.pkl"))]
        random.shuffle(val_paths)
    val_len = cfg.data.n_val if cfg.data.n_val > 0 else len(val_paths)
    val_paths = val_paths[:val_len]
    val_dataset = MicrophysicsCloudDataset(val_paths, data_root_gt_val, n_cam=n_cam,
        rand_cam = rand_cam, mask_type=cfg.ct_net.val_mask_type, mean=mean, std=std,   dataset_name = dataset_name)

    if dataset_name == "BOMEX_pyshdom_varsun_roi":
        test_paths = [f for f in glob.glob(os.path.join(data_root, "cloud*.pkl"))]
        image_indices = [int(cloud_path.split('cloud_results_')[-1].split('.pkl')[0]) for cloud_path in test_paths]
        test_paths = [test_path for ind, test_path in
                     enumerate(test_paths) if np.array(image_indices[ind]) > 6000]
    elif not dataset_name == 'CASS_BOMEX_polarization_pyshdom':
        test_paths = [f for f in glob.glob(os.path.join(data_root, "test/cloud*.pkl"))]
        # test_paths = [os.path.join(data_root, "test/cloud_results_6045.pkl"), os.path.join(data_root, "test/cloud_results_6046.pkl")]
    else:
        test_paths = [f for f in glob.glob(os.path.join(data_root_cass, "test/cloud*.pkl"))]
        test_paths += [f for f in glob.glob(os.path.join(data_root_bomex, "test/cloud*.pkl"))]
        random.shuffle(test_paths)
    test_dataset = MicrophysicsCloudDataset(test_paths, data_root_gt_test, n_cam=n_cam,
                                          rand_cam=rand_cam, mask_type=cfg.ct_net.val_mask_type, mean=mean, std=std,
                                          dataset_name=dataset_name)

    return train_dataset, val_dataset, test_dataset


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
        if self.dataset_name == 'CASS_BOMEX_polarization_pyshdom':
            gt_grid_path = cloud_path
        else:
            gt_grid_path = os.path.join(self.gt_grids_dir, "cloud_results_"+str(image_index)+".pkl")
        projection_path = cloud_path
        with open(cloud_path, 'rb') as f:
            data = pickle.load(f)
        if "at3d" in self.dataset_name and "varwind" in self.dataset_name:
            try:
                with open(gt_grid_path, 'rb') as f:
                    gt_grid_data = pickle.load(f)
            except:
                gt_grid_path = os.path.join(('/').join(self.gt_grids_dir.split('/')[:-1]),'validation', "cloud_results_"+str(image_index)+".pkl")
                with open(gt_grid_path, 'rb') as f:
                    gt_grid_data = pickle.load(f)
        else:
            with open(gt_grid_path, 'rb') as f:
                gt_grid_data = pickle.load(f)
        if "scat" in self.dataset_name:
            if "IQU" in self.dataset_name:
                images = data['images_s']
            elif "IDA" in self.dataset_name:
                images = np.array([data['images'][:,0], data['dolp'], data['aop_s']])
                images = images.transpose([1,0,2,3])
            else:
                NotImplementedError()
        else:
            images = data['images'] #np.concatenate((data['images'], data['DoLPs'][:, None], data['AoLPs'][:, None]), 1)
        mask = None
        if self.mask_type == 'space_carving':
            if ("CASS" in self.dataset_name) and ("pyshdom" in self.dataset_name) and ("CASS" in cloud_path):
                if cloud_path.split("/")[-2] == "validation":
                    direc = "train"
                else:
                    direc = cloud_path.split("/")[-2]
                cloud_path2 = os.path.join('/wdata/roironen/Data/CASS_50m_256x256x139_600CCN/10cameras_20m', direc, cloud_path.split("/")[-1])
                with open(cloud_path2, 'rb') as f:
                    data2 = pickle.load(f)
                mask = data2['mask']
            elif ("BOMEX" in self.dataset_name) and ("pyshdom" in self.dataset_name) and ("varsun" in self.dataset_name) and ("polarization" in self.dataset_name):
                cloud_path2 = glob.glob(os.path.join('/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds', '*',
                                           cloud_path.split("/")[-1]))
                with open(cloud_path2[0], 'rb') as f:
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

        cam_i = torch.arange(self.n_cam)
        if ('varsun' in self.dataset_name) and ('roi' in self.dataset_name):
            index = torch.randperm(20)[0]
            mask = mask[index] if mask is not None else None
            images = images[index]
            images = images[:,None,:,:]
        if 'varying' in self.dataset_name:
            index = torch.randperm(5)[0]
            mask = mask[index] if mask is not None else None
            images = images[index]
            camera_center = data['cameras_pos'][index, cam_i]
            projection_matrix = data['cameras_P'][index, cam_i]
        else:
            camera_center = data['cameras_pos'][cam_i]
            projection_matrix = data['cameras_P'][cam_i]
        if ("CASS" in self.dataset_name) and ("pyshdom" in self.dataset_name) and ("CASS" in cloud_path):
            images = np.squeeze(images[:,cam_i,:,:,:])
        else:
            images = images[cam_i, :, :, :]
        if ('varsun' in self.dataset_name) and ('roi' in self.dataset_name):
            images -= np.array(self.mean)
            images /= np.array(self.std)
        else:
            images -= np.array(self.mean).reshape((1,3,1,1))
            images /= np.array(self.std).reshape((1,3,1,1))

        if "LES" in self.dataset_name:
            lwc_gt = gt_grid_data['lwc_gt']
            veff_gt = np.full_like(lwc_gt,gt_grid_data['veff_gt'])
            veff_gt[lwc_gt == 0] = 0.01
            reff_gt = np.tile(gt_grid_data['reff_gt'][None, None, ...],[lwc_gt.shape[0], lwc_gt.shape[1], 1])
            reff_gt[lwc_gt == 0] = 0
            microphysics = np.array([lwc_gt, reff_gt, veff_gt])
        else:
            microphysics = np.array([gt_grid_data['lwc_gt'],gt_grid_data['reff_gt'],gt_grid_data['veff_gt']])

        if ("CASS" in self.dataset_name) and ("pyshdom" in self.dataset_name) and ("CASS" in cloud_path):
            if microphysics.shape[1] == 74 and microphysics.shape[2] == 74 and microphysics.shape[3] == 37:
                microphysics = microphysics[:,5:-5, 5:-5, :-5]
            assert (microphysics.shape[1] == 64 and microphysics.shape[2] == 64 and microphysics.shape[3] == 32)
        elif ("BOMEX" in self.dataset_name) and ("BOMEX" in cloud_path):
            assert (microphysics.shape[1] == 32 and microphysics.shape[2] == 32 and (microphysics.shape[3] == 32 or microphysics.shape[3] == 64))

        if hasattr(data, 'image_sizes'):
            image_sizes = data['image_sizes'][cam_i]
        else:
            image_sizes = [image.shape[1:] for image in images]

        if "at3d" in self.dataset_name:
            # insert noise to wind_speed value: up to 20% of data['wind_speed']:
            noise_max_val = data['wind_speed']*0.2
            wind_speed_noisy = data['wind_speed'] + np.random.uniform(low=-noise_max_val, high=noise_max_val)
            while wind_speed_noisy<0:
                wind_speed_noisy = data['wind_speed'] + np.random.uniform(low=-noise_max_val, high=noise_max_val)
            env_params = np.array([[wind_speed_noisy, data['sun_zenith'], data['sun_azimuth']]]*images.shape[0])
        elif ("varsun" in self.dataset_name):
            if ("polarization" in self.dataset_name):
                env_params = np.array([[data['sun_zenith'], data['sun_azimuth']]] * images.shape[0])
            else:
                env_params = np.array([[data['sun_zeniths'][index], data['sun_azimuths'][index]]] * images.shape[0])
        else:
            env_params = None

        grid = data['grid']

        if "CASS" in self.dataset_name:
            if grid[0].shape[0] == 74 and grid[1].shape[0] == 74 and grid[2].shape[0] == 37:
                grid[0] = grid[0][5:-5]
                grid[1] = grid[1][5:-5]
                grid[2] = grid[2][:-5]



        return images, microphysics, grid, image_sizes, projection_matrix, camera_center, mask, env_params
