# This file contains a work in progress code.
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

import collections
import os, time
import pickle
import warnings
import glob
# import sys
# sys.path.insert(0, '/home/roironen/pytorch3d/projects/')
import hydra
import numpy as np
import torch
import scipy.io as sio
from VIPCT.visualization import SummaryWriter
from VIPCT.microphysics_dataset import get_cloud_microphysics_datasets, trivial_collate
from VIPCT.CTnet import *
from VIPCT.util.plot_util import *
from omegaconf import OmegaConf
from omegaconf import DictConfig
import matplotlib.pyplot as plt
relative_error = lambda ext_est, ext_gt, eps=1e-6 : torch.norm(ext_est.view(-1) - ext_gt.view(-1),p=1) / (torch.norm(ext_gt.view(-1),p=1) + eps)
mass_error = lambda ext_est, ext_gt, eps=1e-6 : (torch.norm(ext_gt.view(-1),p=1) - torch.norm(ext_est.view(-1),p=1)) / (torch.norm(ext_gt.view(-1),p=1) + eps)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
LWC_EST_TH = 0.01
err_BCE = torch.nn.BCELoss()
import matplotlib
matplotlib.use('TkAgg')

@hydra.main(config_path=CONFIG_DIR, config_name="microphysics_test")
def main(cfg: DictConfig):

    # Set the relevant seeds for reproducibility.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Device on which to run.
    if torch.cuda.is_available() and cfg.debug == False:
        n_device = torch.cuda.device_count()
        cfg.gpu = 0 if n_device==1 else cfg.gpu
        device = f"cuda:{cfg.gpu}"
    else:
        warnings.warn(
            "Please note that although executing on CPU is supported,"
            + "the training is unlikely to finish in reasonable time."
        )
        device = "cpu"
        # Init the visualization visdom env.

    writer = None
    checkpoint_resume_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_resume_path)

    if cfg.save_results:
        log_dir = os.getcwd()
        results_dir = log_dir
        if len(results_dir) > 0:
            # Make the root of the experiment directory.
            # checkpoint_dir = os.path.split(checkpoint_path)
            os.makedirs(results_dir, exist_ok=True)

    resume_cfg_path = os.path.join(checkpoint_resume_path.split('/checkpoints')[0],'.hydra/config.yaml')
    net_cfg = OmegaConf.load(resume_cfg_path)
    cfg = OmegaConf.merge(net_cfg,cfg)
    # DATA_DIR = os.path.join(current_dir, "data")
    _, val_dataset, test_dataset = get_cloud_microphysics_datasets(
        cfg=cfg
    )

    # Initialize the Radiance Field model.
    model = CTnetMicrophysics(cfg=cfg, n_cam=cfg.data.n_cam)

    # Load model
    assert os.path.isfile(checkpoint_resume_path)
    print(f"Resuming from checkpoint {checkpoint_resume_path}.")
    loaded_data = torch.load(checkpoint_resume_path, map_location=device)
    model.load_state_dict(loaded_data["model"])
    model.to(device)

    # Set the model to eval mode.
    model.eval().float()
    model.eval()

    if cfg.choose_thr_from_val_set:
        if 'score' in cfg.optimizer.loss:
            thr_vec = torch.linspace(0, 1, 50, device=device)
            print('choosing threshold based on output mask.')
        else:
            thr_vec = torch.linspace(0, 0.1, 50, device=device)  # use LWC vals
            print('choosing threshold based on LWC values')
        # The test dataloader is just an endless stream of random samples.
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=4,
            collate_fn=trivial_collate,
        )
        # Run the main loop.
        iteration = -1

        # Validation
        F1_score_mat = torch.zeros((len(val_dataloader),len(thr_vec)), device=device)
        val_i = 0
        for val_i, val_batch in enumerate(val_dataloader):
            print('val {}/{}'.format(val_i,len(val_dataloader)))
            # if (val_dataloader.dataset.cloud_dir[val_i]) != '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/test/cloud_results_6065.pkl':
            #     continue
            val_image, microphysics, grid, image_sizes, projection_matrix, camera_center, masks, val_env_params = val_batch  # [0]#.values()
            val_image = torch.tensor(val_image, device=device).float()
            if cfg.data.env_params_num == 0:
                val_env_params = None
            if val_env_params is not None:
                val_env_params = torch.tensor(val_env_params, device=device).float()
                val_env_params = val_env_params[:, :, :cfg.data.env_params_num]
            val_volume = Volumes(torch.tensor(microphysics, device=device).float(), grid)

            val_camera = PerspectiveCameras(image_size=image_sizes,
                                             P=torch.tensor(projection_matrix, device=device).float(),
                                             camera_center=torch.tensor(camera_center, device=device).float(),
                                             device=device)
            if cfg.ct_net.val_mask_type == 'gt_mask':
                masks = val_volume.extinctions > 0  # val_volume._ext_thr
            else:
                masks = [
                    torch.tensor(mask) if mask is not None else torch.ones(*microphysics[0][0].shape, device=device, dtype=bool) for mask in masks]
            masks = [torch.tensor(mask) if mask is not None else mask for mask in masks]

            with torch.no_grad():
                if cfg.optimizer.loss == 'L2_relative_error_w_veff_w_score':
                    est_vols = torch.zeros(int(val_volume.extinctions.numel() * 4 / 3),
                                           device=val_volume.device).reshape(
                        val_volume.extinctions.shape[0], 4, -1)
                else:
                    est_vols = torch.zeros(val_volume.extinctions.numel(), device=val_volume.device).reshape(
                        val_volume.extinctions.shape[0], val_volume.extinctions.shape[1], -1)


                n_points_mask = torch.sum(torch.stack(masks) * 1.0) if isinstance(masks, list) else masks.sum()
                if n_points_mask > cfg.min_mask_points:
                    net_start_time = time.time()

                    val_out = model(
                        val_camera,
                        val_image,
                        val_volume,
                        masks,
                        val_env_params
                    )
                    if val_out['query_indices'] is None:
                        for i, (out_vol, m) in enumerate(zip(val_out["output"], masks)):
                            if m is None:
                                est_vols[i] = out_vol.squeeze(1)
                            else:
                                m = m.view(-1)
                                est_vols[i][m] = out_vol.squeeze(1)
                    else:
                        for est_vol, out_vol, m in zip(est_vols, val_out["output"], val_out['query_indices']):
                            est_vol[:, m] = out_vol.squeeze(1).T  # .reshape(m.shape)[m]
                    time_net = time.time() - net_start_time
                else:
                    time_net = 0
                    continue
                assert len(est_vols) == 1  ##TODO support validation with batch larger than 1

                gt_vol = val_volume.extinctions[0].squeeze()
                gt_lwc = gt_vol[0]
                gt_mask = (gt_lwc > 0)

                est_vols[est_vols < 0] = 0
                if 'score' in cfg.optimizer.loss:
                    est_mask = est_vols[:, 1].squeeze().reshape(gt_mask.shape)
                else:
                    est_mask = est_vols[:, 0].squeeze().reshape(gt_mask.shape)
                for thr_ind, thr in enumerate(thr_vec):
                    est_mask_binary = torch.flatten(est_mask > thr)
                    tp = torch.sum(torch.flatten(gt_mask) & est_mask_binary)
                    fp = torch.sum((torch.flatten(gt_mask)==False) & (est_mask_binary))
                    fn = torch.sum((torch.flatten(gt_mask)) & (est_mask_binary==False))
                    f1_score = 2*tp/(2*tp+fp+fn)
                    F1_score_mat[val_i, thr_ind] = f1_score
        F1_mean_per_thr = torch.mean(F1_score_mat, dim=0)
        MASK_EST_TH = thr_vec[torch.argmax(F1_mean_per_thr)]
    else:
        MASK_EST_TH = 0 #0.2857 #0.2449 #0.5

    print('chosen thr={}'.format(MASK_EST_TH))

    # The test dataloader is just an endless stream of random samples.
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=trivial_collate,
    )


    # Run the main loop.
    iteration = -1
    if writer:
        test_scatter_ind = np.random.permutation(len(test_dataloader))[:5]

    # Validation
    # loss_val = 0
    mask_error_dict = {'true_pos_percent_out_of_gt': [],
                       'true_pos_percent_out_of_mask': []}  # either LWC- or Score-based mask.
    relative_error_dict = {'lwc_full': [],
                           'extinction_full': [],
                           'lwc_with_est_mask': [],
                           'reff_with_est_mask': [],
                           'veff_with_est_mask': [],
                           'extinction_with_est_mask': [],
                           'lwc_with_gt_mask': [],
                           'reff_with_gt_mask': [],
                           'veff_with_gt_mask': [],
                           'extinction_with_gt_mask': [],
                           'lwc_mod': [],
                           'extinction_mod': []
                           }
    relative_mass_error_dict = {'lwc_full': [],
                                'extinction_full': [],
                                'lwc_with_est_mask': [],
                                'reff_with_est_mask': [],
                                'veff_with_est_mask': [],
                                'extinction_with_est_mask': [],
                                'lwc_with_gt_mask': [],
                                'reff_with_gt_mask': [],
                                'veff_with_gt_mask': [],
                                'extinction_with_gt_mask': [],
                                'lwc_mod': [],
                                'extinction_mod': []
                                }
    batch_time_net = []

    reff_vec = []
    veff_vec = []


    test_i = 0
    for test_i, test_batch in enumerate(test_dataloader):
        iteration += 1
        # if (test_dataloader.dataset.cloud_dir[test_i]) != '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/test/cloud_results_6065.pkl':
        #     continue
        test_image, microphysics, grid, image_sizes, projection_matrix, camera_center, masks, test_env_params = test_batch  # [0]#.values()
        test_image = torch.tensor(test_image, device=device).float()
        if cfg.data.env_params_num == 0:
            test_env_params = None
        if test_env_params is not None:
            test_env_params = torch.tensor(test_env_params, device=device).float()
            test_env_params = test_env_params[:, :, :cfg.data.env_params_num]
        test_volume = Volumes(torch.tensor(microphysics, device=device).float(), grid)

        test_camera = PerspectiveCameras(image_size=image_sizes, P=torch.tensor(projection_matrix, device=device).float(),
                                        camera_center=torch.tensor(camera_center, device=device).float(), device=device)
        if cfg.ct_net.test_mask_type == 'gt_mask':
            masks = microphysics[:,0] > 0
        else:
            masks = [torch.tensor(mask) if mask is not None else torch.ones(*microphysics[0][0].shape,device=device, dtype=bool) for mask in masks]
        masks = [torch.tensor(mask) if mask is not None else mask for mask in masks]

        with (torch.no_grad()):
            if cfg.optimizer.loss == 'L2_relative_error_w_veff_w_score':
                est_vols = torch.zeros(int(test_volume.extinctions.numel()*4/3), device=test_volume.device).reshape(
                    test_volume.extinctions.shape[0],4, -1)
            else:
                est_vols = torch.zeros(test_volume.extinctions.numel(), device=test_volume.device).reshape(
                    test_volume.extinctions.shape[0], test_volume.extinctions.shape[1], -1)
            n_points_mask = torch.sum(torch.stack(masks)*1.0) if isinstance(masks, list) else masks.sum()
            if n_points_mask > cfg.min_mask_points:
                net_start_time = time.time()

                test_out = model(
                    test_camera,
                    test_image,
                    test_volume,
                    masks,
                    test_env_params
                )
                if test_out['query_indices'] is None:
                    for i, (out_vol, m) in enumerate(zip(test_out["output"],masks)):
                        if m is None:
                            est_vols[i] = out_vol.squeeze(1)
                        else:
                            m = m.view(-1)
                            est_vols[i][m] = out_vol.squeeze(1)
                else:
                    for est_vol, out_vol, m in zip(est_vols, test_out["output"], test_out['query_indices']):
                        est_vol[:,m]=out_vol.squeeze(1).T#.reshape(m.shape)[m]
                time_net = time.time() - net_start_time
            else:
                time_net = 0
                continue
            assert len(est_vols) == 1  # TODO support validation with batch larger than 1

            gt_vol = test_volume.extinctions[0].squeeze()
            gt_lwc = gt_vol[0]
            gt_mask = (gt_lwc > 0).float()
            gt_reff = gt_vol[1]
            gt_veff = gt_vol[2]
            gt_ext = (3/2)*gt_lwc/gt_reff
            gt_ext[gt_ext != gt_ext] = 0.0  # get rid of nan values
            if cfg.optimizer.loss == 'L2_relative_error_w_veff_w_score':
                est_vols[est_vols < 0] = 0

                est_lwc = est_vols[:,0].squeeze().reshape(gt_lwc.shape)
                est_mask = est_vols[:,1].squeeze().reshape(gt_mask.shape)
                est_reff = est_vols[:,2].squeeze().reshape(gt_reff.shape)
                est_veff = est_vols[:,3].squeeze().reshape(gt_veff.shape)
                est_ext = (3 / 2) * est_lwc / est_reff
                est_ext[est_ext != est_ext] = 0.0  # deals with nan values
                mod_lwc = torch.clone(est_lwc)
                mod_lwc[est_mask <= MASK_EST_TH] = 0.0
                mod_ext = torch.clone(est_ext)
                mod_ext[est_mask <= MASK_EST_TH] = 0.0
            else:
                est_vols = est_vols.squeeze().reshape(gt_vol.shape)
                est_vols[est_vols < 0] = 0

                est_mask = est_vols[0]
                est_lwc = est_vols[0]
                est_reff = est_vols[1]
                est_veff = est_vols[2]
                est_ext = (3 / 2) * est_lwc / est_reff
                est_ext[est_ext != est_ext] = 0.0  # deals with nan values
                mod_lwc = torch.clone(est_lwc)
                mod_ext = torch.clone(est_ext)

            # if (test_dataloader.dataset.cloud_dir[test_i]) == '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/test/cloud_results_6065.pkl':
            #     est_lwc_full = est_lwc.clone().detach()
            #     est_reff_full = est_reff.clone().detach()
            #     est_veff_full = est_veff.clone().detach()
            #     est_lwc_full[est_mask < MASK_EST_TH] = 0
            #     est_lwc_full[est_lwc_full < 0.001] = 0
            #     est_lwc_full[est_lwc_full > 2.5] = 2.5
            #     est_reff_full[est_mask < MASK_EST_TH] = 0
            #     est_reff_full[est_reff_full < 1] = 0
            #     est_reff_full[est_reff_full > 35] = 35
            #     est_veff_full[est_mask < MASK_EST_TH] = 0
            #     est_veff_full[est_veff_full < 0.01] = 0.01  # for SHDOM
            #     est_veff_full[est_veff_full > 0.55] = 0.55
            #
            #     sio.savemat('/wdata/inbalkom/NN_Data/tmp/3d_results_6065.mat', {'est_lwc': est_lwc_full.cpu().numpy(), 'est_reff': est_reff_full.cpu().numpy(),
            #                                                                               'est_veff': est_veff_full.cpu().numpy()})


            print(f'LWC: {relative_error(ext_est=est_lwc, ext_gt=gt_lwc)}, {n_points_mask}')
            print(f'LWC modified: {relative_error(ext_est=mod_lwc, ext_gt=gt_lwc)}, {n_points_mask}')
            print(f'LWC masked: {relative_error(ext_est=est_lwc[est_mask > MASK_EST_TH], ext_gt=gt_lwc[est_mask > MASK_EST_TH])}, {(est_mask > MASK_EST_TH).sum()}')
            print(f'Reff masked: {relative_error(ext_est=est_reff[(est_mask > MASK_EST_TH) & (gt_lwc>0)], ext_gt=gt_reff[(est_mask > MASK_EST_TH) & (gt_lwc>0)])}, {((est_mask > MASK_EST_TH) & (gt_lwc>0)).sum()}')
            print(f'Veff masked: {relative_error(ext_est=est_veff[(est_mask > MASK_EST_TH) & (gt_lwc>0)], ext_gt=gt_veff[(est_mask > MASK_EST_TH) & (gt_lwc>0)])}, {((est_mask > MASK_EST_TH) & (gt_lwc>0)).sum()}')
            print(f'Extinction masked: {relative_error(ext_est=est_ext[est_mask > MASK_EST_TH], ext_gt=gt_ext[est_mask > MASK_EST_TH])}, {(est_mask > MASK_EST_TH).sum()}')
            print(f'Extinction modified: {relative_error(ext_est=mod_ext, ext_gt=gt_ext)}, {n_points_mask}')

            mask_error_dict['true_pos_percent_out_of_gt'].append(((est_mask[gt_mask.bool()] > MASK_EST_TH).sum()/gt_mask.sum()).detach().cpu().numpy())
            mask_error_dict['true_pos_percent_out_of_mask'].append(((est_mask[gt_mask.bool()] > MASK_EST_TH).sum()/((est_mask > MASK_EST_TH).sum()+1e-6)).detach().cpu().numpy())

            relative_error_dict['lwc_full'].append(relative_error(ext_est=est_lwc,ext_gt=gt_lwc).detach().cpu().numpy())
            relative_error_dict['lwc_mod'].append(relative_error(ext_est=mod_lwc, ext_gt=gt_lwc).detach().cpu().numpy())
            relative_error_dict['lwc_with_est_mask'].append(relative_error(ext_est=est_lwc[est_mask > MASK_EST_TH], ext_gt=gt_lwc[est_mask > MASK_EST_TH]).detach().cpu().numpy())
            relative_error_dict['lwc_with_gt_mask'].append(relative_error(ext_est=est_lwc[gt_mask.bool()], ext_gt=gt_lwc[gt_mask.bool()]).detach().cpu().numpy())
            relative_error_dict['reff_with_est_mask'].append(relative_error(ext_est=est_reff[(est_mask > MASK_EST_TH) & (gt_lwc>0)], ext_gt=gt_reff[(est_mask > MASK_EST_TH) & (gt_lwc>0)]).detach().cpu().numpy())
            relative_error_dict['reff_with_gt_mask'].append(relative_error(ext_est=est_reff[gt_mask.bool()], ext_gt=gt_reff[gt_mask.bool()]).detach().cpu().numpy())
            relative_error_dict['veff_with_est_mask'].append(relative_error(ext_est=est_veff[(est_mask > MASK_EST_TH) & (gt_lwc>0)], ext_gt=gt_veff[(est_mask > MASK_EST_TH) & (gt_lwc>0)]).detach().cpu().numpy())
            relative_error_dict['veff_with_gt_mask'].append(relative_error(ext_est=est_veff[gt_mask.bool()], ext_gt=gt_veff[gt_mask.bool()]).detach().cpu().numpy())
            relative_error_dict['extinction_full'].append(relative_error(ext_est=est_ext, ext_gt=gt_ext).detach().cpu().numpy())
            relative_error_dict['extinction_mod'].append(relative_error(ext_est=mod_ext, ext_gt=gt_ext).detach().cpu().numpy())
            relative_error_dict['extinction_with_est_mask'].append(relative_error(ext_est=est_ext[est_mask > MASK_EST_TH], ext_gt=gt_ext[est_mask > MASK_EST_TH]).detach().cpu().numpy())
            relative_error_dict['extinction_with_gt_mask'].append(relative_error(ext_est=est_ext[gt_mask.bool()], ext_gt=gt_ext[gt_mask.bool()]).detach().cpu().numpy())

            relative_mass_error_dict['lwc_full'].append(mass_error(ext_est=est_lwc, ext_gt=gt_lwc).detach().cpu().numpy())
            relative_mass_error_dict['lwc_mod'].append(mass_error(ext_est=mod_lwc, ext_gt=gt_lwc).detach().cpu().numpy())
            relative_mass_error_dict['lwc_with_est_mask'].append(mass_error(ext_est=est_lwc[est_mask > MASK_EST_TH], ext_gt=gt_lwc[est_mask > MASK_EST_TH]).detach().cpu().numpy())
            relative_mass_error_dict['lwc_with_gt_mask'].append(mass_error(ext_est=est_lwc[gt_mask.bool()], ext_gt=gt_lwc[gt_mask.bool()]).detach().cpu().numpy())
            relative_mass_error_dict['reff_with_est_mask'].append(mass_error(ext_est=est_reff[(est_mask > MASK_EST_TH) & (gt_lwc>0)], ext_gt=gt_reff[(est_mask > MASK_EST_TH) & (gt_lwc>0)]).detach().cpu().numpy())
            relative_mass_error_dict['reff_with_gt_mask'].append(mass_error(ext_est=est_reff[gt_mask.bool()], ext_gt=gt_reff[gt_mask.bool()]).detach().cpu().numpy())
            relative_mass_error_dict['veff_with_est_mask'].append(mass_error(ext_est=est_veff[(est_mask > MASK_EST_TH) & (gt_lwc>0)], ext_gt=gt_veff[(est_mask > MASK_EST_TH) & (gt_lwc>0)]).detach().cpu().numpy())
            relative_mass_error_dict['veff_with_gt_mask'].append(mass_error(ext_est=est_veff[gt_mask.bool()], ext_gt=gt_veff[gt_mask.bool()]).detach().cpu().numpy())
            relative_mass_error_dict['extinction_full'].append(mass_error(ext_est=est_ext, ext_gt=gt_ext).detach().cpu().numpy())
            relative_mass_error_dict['extinction_mod'].append(mass_error(ext_est=mod_ext, ext_gt=gt_ext).detach().cpu().numpy())
            relative_mass_error_dict['extinction_with_est_mask'].append(mass_error(ext_est=est_ext[est_mask > MASK_EST_TH], ext_gt=gt_ext[est_mask > MASK_EST_TH]).detach().cpu().numpy())
            relative_mass_error_dict['extinction_with_gt_mask'].append(mass_error(ext_est=est_ext[gt_mask.bool()], ext_gt=gt_ext[gt_mask.bool()]).detach().cpu().numpy())

            batch_time_net.append(time_net)

            cloud_str = test_dataloader.dataset.cloud_dir[test_i].split('_')[-1][:-4]
            # if (est_mask > MASK_EST_TH).sum()>150 and relative_error_dict['lwc_with_est_mask'][-1]<0.3 and \
            #     relative_error_dict['reff_with_est_mask'][-1]<0.3 and relative_error_dict['veff_with_est_mask'][-1]<0.3:
            if 0 :#iteration==6 or iteration==18:
            # if int(cloud_str)==6004 or int(cloud_str)==6293 or int(cloud_str)==6010 or int(cloud_str)==6037 or \
            #     int(cloud_str)==6066 or int(cloud_str)==6473 or int(cloud_str)==6502:
                est_lwc_full = torch.zeros((masks[0].shape[0], masks[0].shape[1], masks[0].shape[2]), device=masks[0].device)
                est_reff_full = torch.zeros((masks[0].shape[0], masks[0].shape[1], masks[0].shape[2]), device=masks[0].device)
                est_veff_full = torch.zeros((masks[0].shape[0], masks[0].shape[1], masks[0].shape[2]), device=masks[0].device)

                est_lwc[est_mask < MASK_EST_TH] = 0
                est_reff[est_mask < MASK_EST_TH] = 0
                est_veff[est_mask < MASK_EST_TH] = 0
                est_ext[est_mask < MASK_EST_TH] = 0

                est_lwc_full = est_lwc
                est_lwc_full[est_lwc_full < 0.001] = 0
                est_lwc_full[est_lwc_full > 2.5] = 2.5
                est_reff_full = est_reff
                est_reff_full[est_reff_full < 1] = 0
                est_reff_full[est_reff_full > 35] = 35
                est_veff_full = est_veff
                est_veff_full[est_veff_full < 0.01] = 0.01  # for SHDOM
                est_veff_full[est_veff_full > 0.55] = 0.55

                time_net = time.time() - net_start_time

                airmspi_cloud = {'cloud_lwc': est_lwc_full.cpu().numpy(), 'cloud_reff': est_reff_full.cpu().numpy(),
                                 'cloud_veff': est_veff_full.cpu().numpy(), 'cloud_ext': est_ext.cpu().numpy(),
                                 'gt_lwc':gt_lwc.cpu().numpy(), 'gt_reff': gt_reff.cpu().numpy(),
                                 'gt_ext': gt_ext.cpu().numpy(), 'gt_veff': gt_veff.cpu().numpy()}
                cloud_str = test_dataloader.dataset.cloud_dir[test_i].split('_')[-1][:-4]
                sio.savemat(os.path.join(results_dir,'bomex_recovery_'+cloud_str+'_new.mat'), airmspi_cloud)

            # if (test_dataloader.dataset.cloud_dir[
            #     test_i]) == '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/test/cloud_results_6359.pkl':
            if 0:  # ((test_i+1)%150)==0: # n_points_mask>200:
                # show_scatter_plot(gt_lwc, est_lwc, 'LWC')
                # show_scatter_plot(gt_ext, est_ext, 'extinction')
                show_scatter_plot(gt_lwc[est_mask > MASK_EST_TH], est_lwc[est_mask > MASK_EST_TH], 'LWC')
                show_scatter_plot(gt_ext[est_mask > MASK_EST_TH], est_ext[est_mask > MASK_EST_TH], 'extinction')
                show_scatter_plot(gt_reff[est_mask > MASK_EST_TH], est_reff[est_mask > MASK_EST_TH],
                                  f'Effective Radius\n output mask',
                                  colorbar_param = est_lwc[est_mask > MASK_EST_TH], colorbar_name = 'LWC')
                show_scatter_plot(gt_veff[est_mask > MASK_EST_TH], est_veff[est_mask > MASK_EST_TH],
                                  f'Effective Variance\n output mask',
                                  colorbar_param = est_lwc[est_mask > MASK_EST_TH], colorbar_name = 'LWC')
                # show_scatter_plot(est_lwc/est_lwc.max(), est_mask,
                #                   f'est lwc mask correlation')
                # show_scatter_plot(gt_lwc/gt_lwc.max(), est_mask,
                #                   f'gt lwc mask correlation')
                volume_plot(gt_lwc>0, (est_mask > MASK_EST_TH).float())

                #show_scatter_plot(gt_lwc,est_lwc, 'LWC')
                #show_scatter_plot_colorbar(gt_lwc, est_lwc, est_mask, f'LWC\n w. score colorbar')
                # show_scatter_plot(gt_lwc, est_lwc, 'LWC')
                # show_scatter_plot(gt_ext, est_ext, 'extinction')
                # show_scatter_plot(gt_reff, est_reff, f'Effective Radius\n no mask')
                # show_scatter_plot(gt_reff[est_lwc>LWC_EST_TH], est_reff[est_lwc>LWC_EST_TH], f'Effective Radius\n Estimated LWC mask')
                # show_scatter_plot(gt_reff[gt_lwc > 0], est_reff[gt_lwc > 0], f'Effective Radius\n GT LWC mask')
                #show_scatter_plot(gt_reff[est_mask > MASK_EST_TH], est_reff[est_mask > MASK_EST_TH], f'Effective Radius\n estimated mask thr')
                #volume_plot(gt_mask, (est_mask > MASK_EST_TH).float())
                # volume_plot(masks[0].float(), (est_lwc>LWC_EST_TH).float())
                # show_scatter_plot(gt_veff, est_veff, 'Effective Variance')
                #show_scatter_plot_altitute(gt_vol,est_vols)
                #volume_plot(gt_lwc,est_lwc)
                #volume_plot(gt_reff, est_reff)
                #volume_plot(gt_veff, est_veff)
                a=5

            if 0:
                num_params = (est_mask > MASK_EST_TH).sum()
                if num_params >= 30:
                    rand_ind = np.random.choice(np.arange(np.array(num_params.cpu())), size=5, replace=False)
                    reff_vec.append(float(est_reff[est_mask > MASK_EST_TH][rand_ind[0]].cpu()))
                    reff_vec.append(float(est_reff[est_mask > MASK_EST_TH][rand_ind[1]].cpu()))
                    reff_vec.append(float(est_reff[est_mask > MASK_EST_TH][rand_ind[2]].cpu()))
                    reff_vec.append(float(est_reff[est_mask > MASK_EST_TH][rand_ind[3]].cpu()))
                    reff_vec.append(float(est_reff[est_mask > MASK_EST_TH][rand_ind[4]].cpu()))
                    veff_vec.append(float(est_veff[est_mask > MASK_EST_TH][rand_ind[0]].cpu()))
                    veff_vec.append(float(est_veff[est_mask > MASK_EST_TH][rand_ind[1]].cpu()))
                    veff_vec.append(float(est_veff[est_mask > MASK_EST_TH][rand_ind[2]].cpu()))
                    veff_vec.append(float(est_veff[est_mask > MASK_EST_TH][rand_ind[3]].cpu()))
                    veff_vec.append(float(est_veff[est_mask > MASK_EST_TH][rand_ind[4]].cpu()))

            if writer:
                writer._iter = iteration
                writer._dataset = 'val'  # .format(test_i)
                if test_i in test_scatter_ind:
                    writer.monitor_scatter_plot(est_vols, gt_vol,ind=test_i)

    if 0:
        countsr, binsr = np.histogram(np.array(reff_vec), density=False)
        # ax.hist(binsr[:-1], binsr, weights=countsr)
        # ax.set_title('re hist')
        countsr = countsr / countsr.sum()

        # fig, ax = plt.subplots()
        countsv, binsv = np.histogram(np.array(veff_vec), density=False)
        # ax.hist(binsv[:-1], binsv, weights=countsv)
        # ax.set_title('ve hist')
        countsv = countsv / countsv.sum()

        count2d, _, _ = np.histogram2d(np.array(reff_vec), np.array(veff_vec), bins=[len(binsr), len(binsv)],
                                       density=False)
        count2d = count2d / count2d.sum()

        mut_info = 0
        r_entropy = 0
        v_entropy = 0
        for indv, countv in enumerate(countsv):
            for indr, countr in enumerate(countsr):
                if countv < 1e-5 or countr < 1e-5 or count2d[indr, indv] < 1e-5:
                    continue
                else:
                    mut_info += count2d[indr, indv] * np.log2((count2d[indr, indv] + 1e-8) / (countv * countr + 1e-8))
                    r_entropy += - countr * np.log2(countr)
                    v_entropy += - countv * np.log2(countv)

        norm_mut_info_bomex_test = mut_info / ((r_entropy + v_entropy) / 2)

        data_train_paths = [f for f in glob.glob(os.path.join(
            "/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/",
            "train/cloud*.pkl"))]
        reff_vec_train = []
        veff_vec_train = []
        for ind, cloud_path in enumerate(data_train_paths):
            print(str(ind + 1) + '/' + str(len(data_train_paths)))
            with open(cloud_path, 'rb') as f:
                data = pickle.load(f)
            lwc_gt = data['lwc_gt']
            reff_gt = data['reff_gt']
            veff_gt = data['veff_gt']
            num_params = (lwc_gt > 0).sum()
            if num_params >= 30:
                rand_ind = np.random.choice(np.arange(num_params), size=3, replace=False)
            else:
                continue
            reff_vec_train.append(reff_gt[lwc_gt > 0][rand_ind[0]])
            reff_vec_train.append(reff_gt[lwc_gt > 0][rand_ind[1]])
            reff_vec_train.append(reff_gt[lwc_gt > 0][rand_ind[2]])
            veff_vec_train.append(veff_gt[lwc_gt > 0][rand_ind[0]])
            veff_vec_train.append(veff_gt[lwc_gt > 0][rand_ind[1]])
            veff_vec_train.append(veff_gt[lwc_gt > 0][rand_ind[2]])


        veff_y_vals = np.arange(0.01, 0.4, 0.001)
        reff_x_vals = np.arange(2.5, 15, 0.01)
        sigma_r = 0.5
        sigma_v = 0.01
        reff_grid, veff_grid = np.meshgrid(reff_x_vals,veff_y_vals)
        rv_test_density = np.zeros([len(veff_y_vals), len(reff_x_vals)])
        for reff, veff in zip(reff_vec, veff_vec):
            rv_test_density += (1/(2*np.pi*sigma_r*sigma_v))*np.exp(-0.5*(reff_grid-reff)**2/sigma_r**2) \
                               * np.exp(-0.5 * (veff_grid - veff)**2 / sigma_v ** 2)*0.001*0.01 / len(reff_vec)



        rv_train_density = np.zeros([len(veff_y_vals), len(reff_x_vals)])
        for reff, veff in zip(reff_vec_train, veff_vec_train):
            rv_train_density += (1 / (2 * np.pi * sigma_r * sigma_v)) * np.exp(-0.5 * (reff_grid - reff) ** 2 / sigma_r ** 2) \
                               * np.exp(-0.5 * (veff_grid - veff) ** 2 / sigma_v ** 2) * 0.001 * 0.01 / len(reff_vec_train)

        norm_kl_div = (rv_train_density*np.log2(rv_train_density/rv_test_density)).sum()/np.log2(reff_grid.size)
        norm_kl_div2 = (rv_test_density * np.log2(rv_test_density/rv_train_density)).sum() / np.log2(reff_grid.size)
        # fig, ax = plt.subplots()
        countsr, binsr = np.histogram(np.array(reff_vec_train), density=False)
        # ax.hist(binsr[:-1], binsr, weights=countsr)
        # ax.set_title('re hist')
        countsr = countsr / countsr.sum()

        # fig, ax = plt.subplots()
        countsv, binsv = np.histogram(np.array(veff_vec_train), density=False)
        # ax.hist(binsv[:-1], binsv, weights=countsv)
        # ax.set_title('ve hist')
        countsv = countsv / countsv.sum()

        count2d, _, _ = np.histogram2d(np.array(reff_vec_train), np.array(veff_vec_train), bins=[len(binsr), len(binsv)],
                                       density=False)
        count2d = count2d / count2d.sum()



        mut_info = 0
        r_entropy = 0
        v_entropy = 0
        for indv, countv in enumerate(countsv):
            for indr, countr in enumerate(countsr):
                if countv < 1e-5 or countr < 1e-5 or count2d[indr, indv] < 1e-5:
                    continue
                else:
                    mut_info += count2d[indr, indv] * np.log2((count2d[indr, indv] + 1e-8) / (countv * countr + 1e-8))
                    r_entropy += - countr * np.log2(countr)
                    v_entropy += - countv * np.log2(countv)

        norm_mut_info_bomex_train = mut_info / ((r_entropy + v_entropy) / 2)

        percent = 0.01
        percent_test = 0.1
        est_param = np.array(veff_vec_train)
        gt_param = np.array(reff_vec_train)
        rho = np.corrcoef(est_param, gt_param)[1, 0]
        num_params = gt_param.size
        num_params_test = np.array(reff_vec).size
        rand_ind = np.random.choice(np.arange(num_params), size=int(percent * num_params), replace=False)
        rand_ind_test = np.random.choice(np.arange(num_params_test), size=int(percent_test * num_params_test), replace=False)
        fig, ax = plt.subplots()
        # ax.set_title(r' ${:1.0f}\%$ randomly sampled; $\rho={:1.2f}$'.format(100 * percent, rho),
        #              fontsize=16)
        ax.set_title(r' ${:1.0f}\%$ randomly sampled'.format(100 * percent),
                     fontsize=16)
        ax.scatter(gt_param[rand_ind], est_param[rand_ind], facecolors='none', edgecolors='b')
        ax.scatter(np.array(reff_vec)[rand_ind_test], np.array(veff_vec)[rand_ind_test],
                   facecolors='none', edgecolors='r')
        ax.set_ylabel('ve', fontsize=14)
        ax.set_xlabel('re', fontsize=14)
        ax.legend([r'BOMEX training set $NMI={:1.3f}$'.format(norm_mut_info_bomex_train),
                   r'BOMEX test set $NMI={:1.3f}$'.format(norm_mut_info_bomex_test)])
        plt.show()


    print('Mask statistics for {} clouds:'.format((test_i + 1)))
    for key, value in mask_error_dict.items():
        value = np.array(value)
        print('    Values for {}: mean={}, std={}'.format(key, np.mean(value), np.std(value)))

    print('Epsilon (relative error) statistics for {} clouds:'.format((test_i + 1)))
    for key, value in relative_error_dict.items():
        value = np.array(value)
        print('    Values for {}: mean={}, std={}'.format(key, np.mean(value), np.std(value)))

    print('Delta (relative mass error) statistics for {} clouds:'.format((test_i + 1)))
    for key, value in relative_mass_error_dict.items():
        value = np.array(value)
        print('    Values for {}: mean={}, std={}'.format(key, np.mean(value), np.std(value)))

    batch_time_net = np.array(batch_time_net)
    print(f'Mean time = {np.mean(batch_time_net)} +- {np.std(batch_time_net)}')

    if cfg.save_results:
        sio.savemat(os.path.join(results_dir, 'numerical_results.mat'), {'mask_errors': mask_error_dict,
                                                                         'relative_errors': relative_error_dict,
                                                                         'relative_mass_errors': relative_mass_error_dict,
                                                                         'batch_time_net': batch_time_net})

    if writer:
        writer._iter = iteration
        writer._dataset = 'val'#.format(test_i)
        writer.monitor_loss(loss_val)
        # writer.monitor_scatterer_error(relative_mass_err, relative_err)
        # writer.monitor_images(test_image)


if __name__ == "__main__":
    main()


