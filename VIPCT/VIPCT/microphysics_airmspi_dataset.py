# This file contains the code for real-world AirMSPI cloud dataset loaders for VIP-CT.
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
ALL_DATASETS_AIRMSPI = ("BOMEX_aux_9cams_polarization", "BOMEX_aux_9cams_polarization_sun_wind",
                        "32N123W_experiment_cloud1", "32N123W_experiment_cloud2", "18S8E_experiment")


def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    batch = np.array(batch, dtype=object).transpose().tolist()
    return batch


def get_microphysics_airmspi_datasets(
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

    if dataset_name not in ALL_DATASETS_AIRMSPI:
        raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")

    if dataset_name == 'BOMEX_aux_9cams_polarization':
        cloud_train_path = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/train') # use 3D clouds from here
        image_root = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256/AIRMSPI_SIMULATIONS') # use push-broom rendered images
        mapping_paths = [f for f in glob.glob(os.path.join(data_root, 'AirMSPI/test/training/voxel_pixel_list*.pkl'))]
        pixel_center_paths = [f for f in glob.glob(os.path.join(data_root, 'AirMSPI/test/training/pixel_centers_*.mat'))]
        image_size = [350, 350]
    elif dataset_name == 'BOMEX_aux_9cams_polarization_wind':
        cloud_train_path = os.path.join(data_root, 'BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/train') # use 3D clouds from here
        image_root = os.path.join('/wdata_visl/inbalkom/NN_Data', 'BOMEX_256x256x100_5000CCN_50m_micro_256/AIRMSPI_SIMULATIONS_AT3D') # use push-broom rendered images
        mapping_paths = [f for f in glob.glob(os.path.join(data_root, 'AirMSPI/test/training/voxel_pixel_list*.pkl'))]
        pixel_center_paths = [f for f in glob.glob(os.path.join(data_root, 'AirMSPI/test/training/pixel_centers_*.mat'))]
        image_size = [350, 350]
    else:
        NotImplementedError()
    ## building map if necessary
    # images_mapping_lists = []
    # pixel_centers_lists = []
    # for mapping_path, pixel_center_path in zip(mapping_paths, pixel_center_paths):
    #     with open(mapping_path, 'rb') as f:
    #         mapping = pickle.load(f)
    #     images_mapping_list = []
    #     pixel_centers_list = []
    #     pixel_centers = sio.loadmat(pixel_center_path)['xpc']
    #     camera_ind = 0
    #     for _, map in mapping.items():
    #         voxels_list = []
    #         pixel_list = []
    #         v = map.values()
    #         voxels = np.array(list(v),dtype=object)
    #         for i, voxel in enumerate(voxels):
    #             if len(voxel)>0:
    #                 pixels = np.unravel_index(voxel, np.array(image_size))
    #                 mean_px = np.mean(pixels,1)
    #                 voxels_list.append(mean_px)
    #                 pixel_list.append(pixel_centers[camera_ind,:,int(mean_px[0]),int(mean_px[1])])
    #             else:
    #                 voxels_list.append([-100000,-100000])
    #                 pixel_list.append([-10000, -10000, -10000])
    #
    #         camera_ind += 1
    #         images_mapping_list.append(voxels_list)
    #         pixel_centers_list.append(pixel_list)
    #     images_mapping_lists.append((images_mapping_list))
    #     pixel_centers_lists.append(pixel_centers_list)
    # print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")
    # with open(os.path.join(data_root, 'AirMSPI/test/training/images_mapping.pkl') as f:
    #     pickle.dump(images_mapping_lists, f, pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(data_root, 'AirMSPI/test/training/pixel_centers.pkl') as f:
    #     pickle.dump(pixel_centers_lists, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join('/wdata/roironen/Data', 'AirMSPI/training/images_mapping.pkl'), 'rb') as f:
        images_mapping_lists = pickle.load(f) # pre-computed voxel-pixel mapping
    with open(os.path.join('/wdata/roironen/Data', 'AirMSPI/training/pixel_centers.pkl'), 'rb') as f:
        pixel_centers_lists = pickle.load(f) # pre-computed 3D pixel center
    if "old" in dataset_name:
        image_train_paths = [f for f in glob.glob(os.path.join(image_root, "without_noise_SIMULATED_AIRMSPI_TRAIN*"))]
    else:
        image_train_paths = [f for f in glob.glob(os.path.join(image_root, "SIMULATED_AIRMSPI_TRAIN*"))]
    image_train_paths = [glob.glob(os.path.join(f, "*.pkl")) for f in image_train_paths]

    assert cfg.data.n_training <= 0

    n_cam = cfg.data.n_cam
    mean = cfg.data.mean
    std = cfg.data.std
    train_dataset = MicrophysicsAirMSPIDataset(
            cloud_train_path,
            image_train_paths,
            mapping=images_mapping_lists,
            n_cam=n_cam,
            mask_type=cfg.ct_net.mask_type,
            mean=mean,
            std=std,
            dataset_name = dataset_name,
            drop_index = cfg.data.drop_index,
            pixel_centers=pixel_centers_lists
        )

    return train_dataset, train_dataset

def get_real_world_microphysics_airmspi_datasets(
    cfg,
    data_root: str = DEFAULT_DATA_ROOT,
) -> Tuple[Dataset]:
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

    if dataset_name not in ALL_DATASETS_AIRMSPI:
        raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")

    if dataset_name == '32N123W_experiment_cloud1':
        image_path = os.path.join(data_root, "AIRMSPI_MEASUREMENTS/corrected_measurements_Inbal_BOMEX.mat")
        data_root2 = '/wdata/roironen/Data'
        mapping_path = os.path.join(data_root2, "AirMSPI/test/32N123W_experiment_cloud1/images_mapping.pkl")
        pixel_centers_path = os.path.join(data_root2, "AirMSPI/test/32N123W_experiment_cloud1/pixel_centers.pkl")
        mask_path = os.path.join(data_root, "AIRMSPI_MEASUREMENTS/mask_72x72x32_vox50x50x40m.mat")
        dx = 0.05
        dy = 0.05
        dz = 0.04
        nx = 72
        ny = 72
        nz = 32
    elif dataset_name == '32N123W_experiment_cloud2':
        image_path = os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/airmspi_9images.mat")
        mapping_path = os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/images_mapping.pkl")
        pixel_centers_path = os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/pixel_centers.pkl")
        mask_path = os.path.join(data_root, "AirMSPI/test/32N123W_experiment_cloud2/mask_60x60x32_vox50x50x40m.mat")
        dx = 0.05
        dy = 0.05
        dz = 0.04
        nx = 60
        ny = 60
        nz = 32
    elif dataset_name == '18S8E_experiment':
        image_path = os.path.join(data_root, "AirMSPI/test/18S8E_experiment/airmspi_9images.mat")
        mapping_path = os.path.join(data_root, "AirMSPI/test/18S8E_experiment/images_mapping.pkl")
        pixel_centers_path = os.path.join(data_root, "AirMSPI/test/18S8E_experiment/pixel_centers.pkl")
        mask_path = os.path.join(data_root, "AirMSPI/test/18S8E_experiment/mask_52x52x32_vox50x50x40m.mat")
        dx = 0.05
        dy = 0.05
        dz = 0.04
        nx = 52
        ny = 52
        nz = 32
    elif dataset_name == '6065_BOMEX_for_test':
        cloud_train_path = os.path.join(data_root,
                                        'BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/train')  # use 3D clouds from here
        image_root = os.path.join(data_root,
                                  'BOMEX_256x256x100_5000CCN_50m_micro_256/AIRMSPI_SIMULATIONS')  # use push-broom rendered images
        mapping_paths = [f for f in glob.glob(os.path.join(data_root, 'AirMSPI/test/training/voxel_pixel_list*.pkl'))]
        pixel_center_paths = [f for f in
                              glob.glob(os.path.join(data_root, 'AirMSPI/test/training/pixel_centers_*.mat'))]
        image_size = [350, 350]
    else:
        NotImplementedError()
    ## building map if necessary
    # images_mapping_lists = []
    # pixel_centers_lists = []
    # for mapping_path, pixel_center_path in zip(mapping_paths, pixel_center_paths):
    #     with open(mapping_path, 'rb') as f:
    #         mapping = pickle.load(f)
    #     images_mapping_list = []
    #     pixel_centers_list = []
    #     pixel_centers = sio.loadmat(pixel_center_path)['xpc']
    #     camera_ind = 0
    #     for _, map in mapping.items():
    #         voxels_list = []
    #         pixel_list = []
    #         v = map.values()
    #         voxels = np.array(list(v),dtype=object)
    #         for i, voxel in enumerate(voxels):
    #             if len(voxel)>0:
    #                 pixels = np.unravel_index(voxel, np.array(image_size))
    #                 mean_px = np.mean(pixels,1)
    #                 voxels_list.append(mean_px)
    #                 pixel_list.append(pixel_centers[camera_ind,:,int(mean_px[0]),int(mean_px[1])])
    #             else:
    #                 voxels_list.append([-100000,-100000])
    #                 pixel_list.append([-10000, -10000, -10000])
    #
    #         camera_ind += 1
    #         images_mapping_list.append(voxels_list)
    #         pixel_centers_list.append(pixel_list)
    #     images_mapping_lists.append((images_mapping_list))
    #     pixel_centers_lists.append(pixel_centers_list)
    # print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")
    # with open('/wdata/roironen/Data/AirMSPI-Varying/training/rebat_images_mapping_lists32x32x32_BOMEX_img350x350.pkl', 'wb') as f:
    #     pickle.dump(images_mapping_lists, f, pickle.HIGHEST_PROTOCOL)
    # with open('/wdata/roironen/Data/AirMSPI-Varying/training/rebat_pixel_centers_lists32x32x32_BOMEX_img350x350.pkl', 'wb') as f:
    #     pickle.dump(pixel_centers_lists, f, pickle.HIGHEST_PROTOCOL)
    with open(mapping_path, 'rb') as f:
        images_mapping_list = pickle.load(f)
    with open(pixel_centers_path, 'rb') as f:
        pixel_centers_list = pickle.load(f)

    # images_mapping_list = [[np.array(map) for map in images_mapping_list]]
    # pixel_centers_list = [[np.array(centers) for centers in pixel_centers_list]]

    gx = np.linspace(0, dx * nx, nx, dtype=np.float32)
    gy = np.linspace(0, dy * ny, ny, dtype=np.float32)
    gz = np.linspace(0, dz * nz, nz, dtype=np.float32)
    grid = [np.array([gx, gy, gz])]
    mask = sio.loadmat(mask_path)['mask']

    assert cfg.data.n_training <= 0

    n_cam = cfg.data.n_cam
    mean = cfg.data.mean
    std = cfg.data.std
    dataset = MicrophysicsAirMSPIDataset_test(
        image_dir = image_path,
        mapping=images_mapping_list,
        n_cam=n_cam,
        mask_type=cfg.ct_net.mask_type,
        mean=mean,
        std=std,
        dataset_name = dataset_name,
        drop_index = cfg.data.drop_index,
        pixel_centers=pixel_centers_list,
        mask = mask,
        grid = grid

    )

    return dataset

class MicrophysicsAirMSPIDataset(Dataset):
    def __init__(self, cloud_dir,image_dir, n_cam,mapping, pixel_centers, mask_type=None, mean=0, std=1, dataset_name='', drop_index=-1):
        self.cloud_dir = cloud_dir
        self.mapping = mapping
        self.image_dir = image_dir
        self.mask_type = mask_type
        self.n_cam = n_cam
        self.mean = mean
        self.std = std
        self.dataset_name = dataset_name
        self.pixel_centers = pixel_centers
        self.drop_index = drop_index
        if self.n_cam != 9 and self.drop_index>-1:
            for map in self.mapping:
                map.pop(drop_index)
            self.pixel_centers = np.delete(self.pixel_centers,self.drop_index,1)

    def __len__(self):
        return len(self.cloud_dir)

    def __getitem__(self, idx):
        geometry_ind = np.random.randint(len(self.image_dir))
        image_dir = self.image_dir[geometry_ind][idx]
        image_index = image_dir.split('cloud_results_')[-1].split('.pkl')[0]
        cloud_path = os.path.join(self.cloud_dir, f"cloud_results_{image_index}.pkl")

        try:
            with open(image_dir, 'rb') as f:
                image_data = pickle.load(f)
                images = image_data['images']
                env_params = []
                if 'sun_zenith' in image_data.keys():
                    env_params.append(image_data['sun_zenith'])
                if 'sun_azimuth' in image_data.keys():
                    env_params.append(image_data['sun_azimuth'])
                if 'wind_speed' in image_data.keys():
                    env_params.append(image_data['wind_speed'])
            with open(cloud_path, 'rb') as f:
                data = pickle.load(f)

            if self.n_cam != 9:
                images = np.delete(images, self.drop_index,0)
            # if len(env_params):
            #     # add env_params as channels to all of the images
            #     env_params = np.array(env_params)
            #     env_params = np.expand_dims(env_params, axis=[0, 2, 3])
            #     env_params_images = np.tile(env_params, [images.shape[0], 1, images.shape[2], images.shape[3]])
            #     images = np.concatenate((images, env_params_images), axis=1)
            mask = None
            if self.mask_type == 'space_carving':
                mask = image_data['mask']
            elif self.mask_type == 'space_carving_morph':
                mask = image_data['mask_morph']
            elif self.mask_type == 'gt_mask':
                mask = (data['lwc_gt'] > 0) * (data['reff_gt'] > 0) * (data['veff_gt'] > 0)
            images -= np.array(self.mean).reshape((1, 3, 1, 1))
            images /= np.array(self.std).reshape((1, 3, 1, 1))
            grid = data['grid']
            microphysics = np.array([data['lwc_gt'],data['reff_gt'],data['veff_gt']]) #np.array([data['lwc_gt']/10,data['reff_gt'],data['veff_gt']]) # convert BOMEX clouds to BOMEX_aux clouds

            images_mapping_list = [np.array(map)[mask.ravel()] for map in self.mapping[geometry_ind]]
            pixel_centers = [np.array(centers)[mask.ravel()] for centers in self.pixel_centers[geometry_ind]]
        except:
            images = np.array(-1)
            microphysics = -1
            grid = -1
            images_mapping_list = -1
            pixel_centers = -1
            mask = -1

        return images, microphysics, grid, images_mapping_list, pixel_centers, mask

class MicrophysicsAirMSPIDataset_test(Dataset):
    def __init__(self, image_dir, n_cam, mapping, pixel_centers, mask_type=None, mean=0, std=1, dataset_name='',
                 drop_index=-1, mask = None, grid = None):
        self.mapping = mapping
        self.image_dir = image_dir
        self.mask_type = mask_type
        self.n_cam = n_cam
        self.mean = mean
        self.std = std
        self.dataset_name = dataset_name
        self.pixel_centers = pixel_centers
        self.drop_index = drop_index
        self.mask = mask
        self.grid =grid
        if self.n_cam != 9 and self.drop_index>-1:
            self.mapping.pop(self.drop_index)
            self.pixel_centers = np.delete(self.pixel_centers,self.drop_index,0)


    def __getitem__(self, idx):
        images = sio.loadmat(self.image_dir)['croped_airmspi_images']
        images = np.rollaxis(images, -1, 1)
        if self.n_cam != 9:
            images = np.delete(images, self.drop_index, 0)
        mask = None
        if self.mask_type == 'space_carving':
            mask = (self.mask).astype(bool)
        else:
            mask = np.ones_like(self.grid)
        images -= np.array(self.mean).reshape((1, 3, 1, 1))
        images /= np.array(self.std).reshape((1, 3, 1, 1))
        images_mapping_list = [[np.array(map) for map in self.mapping]]
        pixel_centers_list = [[np.array(centers) for centers in self.pixel_centers]]

        return images, self.grid, images_mapping_list, pixel_centers_list, mask
