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
MASK_EST_TH = 0.5
err_BCE = torch.nn.BCELoss()

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
    log_dir = os.getcwd()
    writer = None #SummaryWriter(log_dir)
    results_dir = log_dir #os.path.join(log_dir, 'test_results')
    checkpoint_resume_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_resume_path)
    if len(results_dir) > 0:
        # Make the root of the experiment directory.
        # checkpoint_dir = os.path.split(checkpoint_path)
        os.makedirs(results_dir, exist_ok=True)

    resume_cfg_path = os.path.join(checkpoint_resume_path.split('/checkpoints')[0],'.hydra/config.yaml')
    net_cfg = OmegaConf.load(resume_cfg_path)
    cfg=OmegaConf.merge(net_cfg,cfg)
    # DATA_DIR = os.path.join(current_dir, "data")
    _, val_dataset = get_cloud_microphysics_datasets(
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

    # The validation dataloader is just an endless stream of random samples.
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=trivial_collate,
    )

    # Set the model to eval mode.
    model.eval().float()
    model.eval()

    # Run the main loop.
    iteration = -1
    if writer:
        val_scatter_ind = np.random.permutation(len(val_dataloader))[:5]

    # Validation
    # loss_val = 0
    lwc_mask_true_positive_precent_out_of_gt = []
    lwc_mask_true_positive_precent_out_of_mask = []
    lwc_relative_err= []
    lwc_relative_err_lwc_est_th = []
    lwc_relative_err_lwc_gt_th = []
    reff_relative_err = []
    reff_relative_err_lwc_est_th = []
    reff_relative_err_lwc_gt_th= []
    veff_relative_err= []
    veff_relative_err_lwc_est_th = []
    veff_relative_err_lwc_gt_th = []
    lwc_relative_mass_err = []
    lwc_relative_mass_err_lwc_est_th = []
    lwc_relative_mass_err_lwc_gt_th = []
    reff_relative_mass_err = []
    reff_relative_mass_err_lwc_est_th = []
    reff_relative_mass_err_lwc_gt_th = []
    veff_relative_mass_err = []
    veff_relative_mass_err_lwc_est_th = []
    veff_relative_mass_err_lwc_gt_th = []
    batch_time_net = []
    if cfg.optimizer.lwc_loss == 'L2_relative_error_w_veff_w_score':
        mask_bce_err = []
        mask_true_positive_precent_out_of_gt = []
        mask_true_positive_precent_out_of_mask = []
        lwc_relative_err_est_mask_th = []
        lwc_relative_mass_err_est_mask_th = []
        reff_relative_err_est_mask_th = []
        reff_relative_mass_err_est_mask_th = []
        veff_relative_err_est_mask_th = []
        veff_relative_mass_err_est_mask_th = []
    val_i = 0
    for val_i, val_batch in enumerate(val_dataloader):
        if (val_dataloader.dataset.cloud_dir[val_i]) != '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/test/cloud_results_6065.pkl':
            continue
        val_image, microphysics, grid, image_sizes, projection_matrix, camera_center, masks = val_batch  # [0]#.values()
        val_image = torch.tensor(val_image, device=device).float()
        val_volume = Volumes(torch.tensor(microphysics, device=device).float(), grid)

        val_camera = PerspectiveCameras(image_size=image_sizes, P=torch.tensor(projection_matrix, device=device).float(),
                                        camera_center=torch.tensor(camera_center, device=device).float(), device=device)
        if model.val_mask_type == 'gt_mask':
            masks = val_volume.extinctions > 0 #val_volume._ext_thr
        else:
            masks = [torch.tensor(mask) if mask is not None else torch.ones(*microphysics[0][0].shape,device=device, dtype=bool) for mask in masks]
        masks = [torch.tensor(mask) if mask is not None else mask for mask in masks]

        with torch.no_grad():
            if cfg.optimizer.lwc_loss == 'L2_relative_error_w_veff_w_score':
                est_vols = torch.zeros(int(val_volume.extinctions.numel()*4/3), device=val_volume.device).reshape(
                    val_volume.extinctions.shape[0],4, -1)
            else:
                est_vols = torch.zeros(val_volume.extinctions.numel(), device=val_volume.device).reshape(
                    val_volume.extinctions.shape[0], val_volume.extinctions.shape[1], -1)
            n_points_mask = torch.sum(torch.stack(masks)*1.0) if isinstance(masks, list) else masks.sum()
            if n_points_mask > cfg.min_mask_points:
                net_start_time = time.time()

                val_out = model(
                    val_camera,
                    val_image,
                    val_volume,
                    masks
                )
                if val_out['query_indices'] is None:
                    for i, (out_vol, m) in enumerate(zip(val_out["output"],masks)):
                        if m is None:
                            est_vols[i] = out_vol.squeeze(1)
                        else:
                            m = m.view(-1)
                            est_vols[i][m] = out_vol.squeeze(1)
                else:
                    for est_vol, out_vol, m in zip(est_vols, val_out["output"], val_out['query_indices']):
                        est_vol[:,m]=out_vol.squeeze(1).T#.reshape(m.shape)[m]
                time_net = time.time() - net_start_time
            else:
                time_net = 0
                continue
            assert len(est_vols)==1 ##TODO support validation with batch larger than 1

            gt_vol = val_volume.extinctions[0].squeeze()
            gt_lwc = gt_vol[0]
            gt_reff = gt_vol[1]
            gt_veff = gt_vol[2]

            if cfg.optimizer.lwc_loss == 'L2_relative_error_w_veff_w_score':
                # mask_bce_err = []
                # mask_relative_err = []
                gt_mask = (gt_lwc > 0).float()

                est_vols[est_vols < 0] = 0

                est_lwc = est_vols[:,0].squeeze().reshape(gt_lwc.shape)
                est_mask = est_vols[:,1].squeeze().reshape(gt_mask.shape)
                est_reff = est_vols[:,2].squeeze().reshape(gt_reff.shape)
                est_veff = est_vols[:,3].squeeze().reshape(gt_veff.shape)
            else:

                est_vols = est_vols.squeeze().reshape(gt_vol.shape)
                # est_vols[gt_vol==0] = 0
                est_vols[est_vols<0] = 0

                est_lwc = est_vols[0]
                est_reff = est_vols[1]
                est_veff = est_vols[2]

            if (val_dataloader.dataset.cloud_dir[val_i]) == '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/test/cloud_results_6065.pkl':
                est_lwc_full = est_lwc.clone().detach()
                est_reff_full = est_reff.clone().detach()
                est_veff_full = est_veff.clone().detach()
                est_lwc_full[est_mask < MASK_EST_TH] = 0
                est_lwc_full[est_lwc_full < 0.001] = 0
                est_lwc_full[est_lwc_full > 2.5] = 2.5
                est_reff_full[est_mask < MASK_EST_TH] = 0
                est_reff_full[est_reff_full < 1] = 0
                est_reff_full[est_reff_full > 35] = 35
                est_veff_full[est_mask < MASK_EST_TH] = 0
                est_veff_full[est_veff_full < 0.01] = 0.01  # for SHDOM
                est_veff_full[est_veff_full > 0.55] = 0.55

                sio.savemat('/wdata/inbalkom/NN_Data/tmp/3d_results_6065.mat', {'est_lwc': est_lwc_full.cpu().numpy(), 'est_reff': est_reff_full.cpu().numpy(),
                                                                                          'est_veff': est_veff_full.cpu().numpy()})


            print(f'LWC: {relative_error(ext_est=est_lwc,ext_gt=gt_lwc)}, {n_points_mask}')
            print(f'Reff est lwc mask: {relative_error(ext_est=est_reff[est_lwc>LWC_EST_TH],ext_gt=gt_reff[est_lwc>LWC_EST_TH])}, {(est_lwc>LWC_EST_TH).sum()}')
            print(f'Reff gt lwc mask: {relative_error(ext_est=est_reff[gt_lwc > 0], ext_gt=gt_reff[gt_lwc > 0])}, {(gt_lwc > 0).sum()}')
            print(f'Veff est lwc mask: {relative_error(ext_est=est_veff[est_lwc > LWC_EST_TH], ext_gt=gt_veff[est_lwc > LWC_EST_TH])}, {(est_lwc > LWC_EST_TH).sum()}')
            print(f'Veff gt lwc mask: {relative_error(ext_est=est_veff[gt_lwc > 0], ext_gt=gt_veff[gt_lwc > 0])}, {(gt_lwc > 0).sum()}')
            if cfg.optimizer.lwc_loss == 'L2_relative_error_w_veff_w_score':
                print(f'Mask BCE err: {err_BCE(est_mask, gt_mask)}, {n_points_mask}')
                print(f'Reff score mask: {relative_error(ext_est=est_reff[est_mask > MASK_EST_TH], ext_gt=gt_reff[est_mask > MASK_EST_TH])}, {(est_mask > MASK_EST_TH).sum()}')
                print(f'Veff score mask: {relative_error(ext_est=est_veff[est_mask > MASK_EST_TH], ext_gt=gt_veff[est_mask > MASK_EST_TH])}, {(est_mask > MASK_EST_TH).sum()}')

                lwc_mask_true_positive_precent_out_of_gt.append(((est_lwc[gt_mask.bool()] > LWC_EST_TH).sum()/gt_mask.sum()).detach().cpu().numpy())
                lwc_mask_true_positive_precent_out_of_mask.append(((est_lwc[gt_mask.bool()] > LWC_EST_TH).sum() / (est_lwc > LWC_EST_TH).sum()).detach().cpu().numpy())
            lwc_relative_err.append(relative_error(ext_est=est_lwc,ext_gt=gt_lwc).detach().cpu().numpy())#torch.norm(val_out["output"] - val_out["volume"], p=1) / (torch.norm(val_out["volume"], p=1) + 1e-6)
            lwc_relative_err_lwc_est_th.append(relative_error(ext_est=est_lwc[est_lwc>LWC_EST_TH],ext_gt=gt_lwc[est_lwc>LWC_EST_TH]).detach().cpu().numpy())
            lwc_relative_err_lwc_gt_th.append(relative_error(ext_est=est_lwc[gt_lwc > 0], ext_gt=gt_lwc[gt_lwc > 0]).detach().cpu().numpy())
            reff_relative_err.append(relative_error(ext_est=est_reff,ext_gt=gt_reff).detach().cpu().numpy())
            reff_relative_err_lwc_est_th.append(relative_error(ext_est=est_reff[est_lwc>LWC_EST_TH],ext_gt=gt_reff[est_lwc>LWC_EST_TH]).detach().cpu().numpy())
            reff_relative_err_lwc_gt_th.append(relative_error(ext_est=est_reff[gt_lwc > 0], ext_gt=gt_reff[gt_lwc > 0]).detach().cpu().numpy())
            veff_relative_err.append(relative_error(ext_est=est_veff,ext_gt=gt_veff).detach().cpu().numpy())#torch.norm(val_out["output"] - val_out["volume"], p=1) / (torch.norm(val_out["volume"], p=1) + 1e-6)
            veff_relative_err_lwc_est_th.append(relative_error(ext_est=est_veff[est_lwc>LWC_EST_TH],ext_gt=gt_veff[est_lwc>LWC_EST_TH]).detach().cpu().numpy())  # torch.norm(val_out["output"] - val_out["volume"], p=1) / (torch.norm(val_out["volume"], p=1) + 1e-6)
            veff_relative_err_lwc_gt_th.append(relative_error(ext_est=est_veff[gt_lwc > 0],ext_gt=gt_veff[gt_lwc > 0]).detach().cpu().numpy())  # torch.norm(val_out["output"] - val_out["volume"], p=1) / (torch.norm(val_out["volume"], p=1) + 1e-6)
            lwc_relative_mass_err.append(mass_error(ext_est=est_lwc,ext_gt=gt_lwc).detach().cpu().numpy())
            lwc_relative_mass_err_lwc_est_th.append(mass_error(ext_est=est_lwc[est_lwc > LWC_EST_TH], ext_gt=gt_lwc[est_lwc > LWC_EST_TH]).detach().cpu().numpy())  # (torch.norm(val_out["output"], p=1) - torch.norm(val_out["volume"], p=1)) / (torch.norm(val_out["volume"], p=1) + 1e-6)
            lwc_relative_mass_err_lwc_gt_th.append(mass_error(ext_est=est_lwc[gt_lwc > 0], ext_gt=gt_lwc[gt_lwc > 0]).detach().cpu().numpy())
            reff_relative_mass_err.append(mass_error(ext_est=est_reff, ext_gt=gt_reff).detach().cpu().numpy())
            reff_relative_mass_err_lwc_est_th.append(mass_error(ext_est=est_reff[est_lwc>LWC_EST_TH],ext_gt=gt_reff[est_lwc>LWC_EST_TH]).detach().cpu().numpy())#(torch.norm(val_out["output"], p=1) - torch.norm(val_out["volume"], p=1)) / (torch.norm(val_out["volume"], p=1) + 1e-6)
            reff_relative_mass_err_lwc_gt_th.append(mass_error(ext_est=est_reff[gt_lwc > 0],ext_gt=gt_reff[gt_lwc > 0]).detach().cpu().numpy())#(torch.norm(val_out["output"], p=1) - torch.norm(val_out["volume"], p=1)) / (torch.norm(val_out["volume"], p=1) + 1e-6)
            veff_relative_mass_err.append(mass_error(ext_est=est_veff,ext_gt=gt_veff).detach().cpu().numpy())#(torch.norm(val_out["output"], p=1) - torch.norm(val_out["volume"], p=1)) / (torch.norm(val_out["volume"], p=1) + 1e-6)
            veff_relative_mass_err_lwc_est_th.append(mass_error(ext_est=est_veff[est_lwc > LWC_EST_TH], ext_gt=gt_veff[est_lwc > LWC_EST_TH]).detach().cpu().numpy())
            veff_relative_mass_err_lwc_gt_th.append(mass_error(ext_est=est_veff[gt_lwc > 0], ext_gt=gt_veff[gt_lwc > 0]).detach().cpu().numpy())
            batch_time_net.append(time_net)

            if cfg.optimizer.lwc_loss == 'L2_relative_error_w_veff_w_score':
                mask_bce_err.append(err_BCE(est_mask, gt_mask).detach().cpu().numpy())
                mask_true_positive_precent_out_of_gt.append(((est_mask[gt_mask.bool()] > MASK_EST_TH).sum()/gt_mask.sum()).detach().cpu().numpy())
                mask_true_positive_precent_out_of_mask.append(((est_mask[gt_mask.bool()] > MASK_EST_TH).sum()/(est_mask > MASK_EST_TH).sum()).detach().cpu().numpy())
                lwc_relative_err_est_mask_th.append(relative_error(ext_est=est_lwc[est_mask > MASK_EST_TH],ext_gt=gt_lwc[est_mask > MASK_EST_TH]).detach().cpu().numpy())
                lwc_relative_mass_err_est_mask_th.append(mass_error(ext_est=est_lwc[est_mask > MASK_EST_TH],ext_gt=gt_lwc[est_mask > MASK_EST_TH]).detach().cpu().numpy())
                reff_relative_err_est_mask_th.append(relative_error(ext_est=est_reff[est_mask > MASK_EST_TH],ext_gt=gt_reff[est_mask > MASK_EST_TH]).detach().cpu().numpy())
                reff_relative_mass_err_est_mask_th.append(mass_error(ext_est=est_reff[est_mask > MASK_EST_TH],ext_gt=gt_reff[est_mask > MASK_EST_TH]).detach().cpu().numpy())
                veff_relative_err_est_mask_th.append(relative_error(ext_est=est_veff[est_mask > MASK_EST_TH],ext_gt=gt_veff[est_mask > MASK_EST_TH]).detach().cpu().numpy())
                veff_relative_mass_err_est_mask_th.append(mass_error(ext_est=est_veff[est_mask > MASK_EST_TH],ext_gt=gt_veff[est_mask > MASK_EST_TH]).detach().cpu().numpy())

            if 1: # ((val_i+1)%150)==0: # n_points_mask>200: #True: #False:
                #show_scatter_plot(gt_lwc,est_lwc, 'LWC')
                #show_scatter_plot_colorbar(gt_lwc, est_lwc, est_mask, f'LWC\n w. score colorbar')
                show_scatter_plot(gt_lwc, est_lwc, 'LWC')
                show_scatter_plot(gt_reff, est_reff, f'Effective Radius\n no mask')
                show_scatter_plot(gt_reff[est_lwc>LWC_EST_TH], est_reff[est_lwc>LWC_EST_TH], f'Effective Radius\n Estimated LWC mask')
                show_scatter_plot(gt_reff[gt_lwc > 0], est_reff[gt_lwc > 0], f'Effective Radius\n GT LWC mask')
                #show_scatter_plot(gt_reff[est_mask > MASK_EST_TH], est_reff[est_mask > MASK_EST_TH], f'Effective Radius\n estimated mask thr')
                #volume_plot(gt_mask, (est_mask > MASK_EST_TH).float())
                volume_plot(masks[0].float(), (est_lwc>LWC_EST_TH).float())
                show_scatter_plot(gt_veff, est_veff, 'Effective Variance')
                #show_scatter_plot_altitute(gt_vol,est_vols)
                #volume_plot(gt_lwc,est_lwc)
                #volume_plot(gt_reff, est_reff)
                #volume_plot(gt_veff, est_veff)
                a=5
            if writer:
                writer._iter = iteration
                writer._dataset = 'val'  # .format(val_i)
                if val_i in val_scatter_ind:
                    writer.monitor_scatter_plot(est_vols, gt_vol,ind=val_i)


    lwc_relative_err = np.array(lwc_relative_err)
    lwc_relative_err_lwc_est_th = np.array(lwc_relative_err_lwc_est_th)
    lwc_relative_err_lwc_gt_th = np.array(lwc_relative_err_lwc_gt_th)
    reff_relative_err = np.array(reff_relative_err)
    reff_relative_err_lwc_est_th = np.array(reff_relative_err_lwc_est_th)
    reff_relative_err_lwc_gt_th = np.array(reff_relative_err_lwc_gt_th)
    veff_relative_err = np.array(veff_relative_err)
    veff_relative_err_lwc_est_th = np.array(veff_relative_err_lwc_est_th)
    veff_relative_err_lwc_gt_th = np.array(veff_relative_err_lwc_gt_th)
    lwc_relative_mass_err = np.array(lwc_relative_mass_err)
    lwc_relative_mass_err_lwc_est_th = np.array(lwc_relative_mass_err_lwc_est_th)
    lwc_relative_mass_err_lwc_gt_th = np.array(lwc_relative_mass_err_lwc_gt_th)
    reff_relative_mass_err = np.array(reff_relative_mass_err)
    reff_relative_mass_err_lwc_est_th = np.array(reff_relative_mass_err_lwc_est_th)
    reff_relative_mass_err_lwc_gt_th = np.array(reff_relative_mass_err_lwc_gt_th)
    veff_relative_mass_err = np.array(veff_relative_mass_err)
    veff_relative_mass_err_lwc_est_th = np.array(veff_relative_mass_err_lwc_est_th)
    veff_relative_mass_err_lwc_gt_th = np.array(veff_relative_mass_err_lwc_gt_th)
    batch_time_net = np.array(batch_time_net)
    if cfg.optimizer.lwc_loss == 'L2_relative_error_w_veff_w_score':
        lwc_mask_true_positive_precent_out_of_gt = np.array(lwc_mask_true_positive_precent_out_of_gt)
        lwc_mask_true_positive_precent_out_of_mask = np.array(lwc_mask_true_positive_precent_out_of_mask)

        mask_bce_err = np.array(mask_bce_err)
        mask_true_positive_precent_out_of_gt = np.array(mask_true_positive_precent_out_of_gt)
        mask_true_positive_precent_out_of_mask = np.array(mask_true_positive_precent_out_of_mask)
        lwc_relative_err_est_mask_th = np.array(lwc_relative_err_est_mask_th)
        lwc_relative_mass_err_est_mask_th = np.array(lwc_relative_mass_err_est_mask_th)
        reff_relative_err_est_mask_th = np.array(reff_relative_err_est_mask_th)
        reff_relative_mass_err_est_mask_th = np.array(reff_relative_mass_err_est_mask_th)
        veff_relative_err_est_mask_th = np.array(veff_relative_err_est_mask_th)
        veff_relative_mass_err_est_mask_th = np.array(veff_relative_mass_err_est_mask_th)


    print(f'est LWC Mask: true positive precent out of gt {np.mean(lwc_mask_true_positive_precent_out_of_gt)} with std of {np.std(lwc_mask_true_positive_precent_out_of_gt)} for {(val_i + 1)} clouds')
    print(f'est LWC Mask: true positive precent out of mask {np.mean(lwc_mask_true_positive_precent_out_of_mask)} with std of {np.std(lwc_mask_true_positive_precent_out_of_mask)} for {(val_i + 1)} clouds')
    print(f'LWC: mean relative error {np.mean(lwc_relative_err)} with std of {np.std(lwc_relative_err)} for {(val_i + 1)} clouds')
    print(f'LWC est lwc mask: mean relative error {np.mean(lwc_relative_err_lwc_est_th)} with std of {np.std(lwc_relative_err_lwc_est_th)} for {(val_i + 1)} clouds')
    print(f'LWC gt lwc mask: mean relative error {np.mean(lwc_relative_err_lwc_gt_th)} with std of {np.std(lwc_relative_err_lwc_gt_th)} for {(val_i + 1)} clouds')
    print(f'Reff: mean relative error {np.mean(reff_relative_err)} with std of {np.std(reff_relative_err)} for {(val_i + 1)} clouds')
    print(f'Reff est lwc mask: mean relative error {np.mean(reff_relative_err_lwc_est_th)} with std of {np.std(reff_relative_err_lwc_est_th)} for {(val_i + 1)} clouds')
    print(f'Reff gt lwc mask: mean relative error {np.mean(reff_relative_err_lwc_gt_th)} with std of {np.std(reff_relative_err_lwc_gt_th)} for {(val_i + 1)} clouds')
    print(f'Veff: mean relative error {np.mean(veff_relative_err)} with std of {np.std(veff_relative_err)} for {(val_i + 1)} clouds')
    print(f'Veff est lwc mask: mean relative error {np.mean(veff_relative_err_lwc_est_th)} with std of {np.std(veff_relative_err_lwc_est_th)} for {(val_i + 1)} clouds')
    print(f'Veff gt lwc mask: mean relative error {np.mean(veff_relative_err_lwc_gt_th)} with std of {np.std(veff_relative_err_lwc_gt_th)} for {(val_i + 1)} clouds')
    print(f'LWC: mean relative mass error {np.mean(lwc_relative_mass_err)} with std of {np.std(lwc_relative_mass_err)} for {(val_i + 1)} clouds')
    print(f'LWC est lwc mask: mean relative mass error {np.mean(lwc_relative_mass_err_lwc_est_th)} with std of {np.std(lwc_relative_mass_err_lwc_est_th)} for {(val_i + 1)} clouds')
    print(f'LWC gt lwc mask: mean relative mass error {np.mean(lwc_relative_mass_err_lwc_gt_th)} with std of {np.std(lwc_relative_mass_err_lwc_gt_th)} for {(val_i + 1)} clouds')
    print(f'Reff: mean relative mass error {np.mean(reff_relative_mass_err)} with std of {np.std(reff_relative_mass_err)} for {(val_i + 1)} clouds')
    print(f'Reff est lwc mask: mean relative mass error {np.mean(reff_relative_mass_err_lwc_est_th)} with std of {np.std(reff_relative_mass_err_lwc_est_th)} for {(val_i + 1)} clouds')
    print(f'Reff gt lwc mask: mean relative mass error {np.mean(reff_relative_mass_err_lwc_gt_th)} with std of {np.std(reff_relative_mass_err_lwc_gt_th)} for {(val_i + 1)} clouds')
    print(f'Veff: mean relative mass error {np.mean(veff_relative_mass_err)} with std of {np.std(veff_relative_mass_err)} for {(val_i + 1)} clouds')
    print(f'Veff est lwc mask: mean relative mass error {np.mean(veff_relative_mass_err_lwc_est_th)} with std of {np.std(veff_relative_mass_err_lwc_est_th)} for {(val_i + 1)} clouds')
    print(f'Veff gt lwc mask: mean relative mass error {np.mean(veff_relative_mass_err_lwc_gt_th)} with std of {np.std(veff_relative_mass_err_lwc_gt_th)} for {(val_i + 1)} clouds')
    print(f'Mean time = {np.mean(batch_time_net)} +- {np.std(batch_time_net)}')
    if cfg.optimizer.lwc_loss == 'L2_relative_error_w_veff_w_score':
        print(f'Mask: BCE error {np.mean(mask_bce_err)} with std of {np.std(mask_bce_err)} for {(val_i + 1)} clouds')
        print(f'Mask: true positive precent out of gt {np.mean(mask_true_positive_precent_out_of_gt)} with std of {np.std(mask_true_positive_precent_out_of_gt)} for {(val_i + 1)} clouds')
        print(f'Mask: true positive precent out of mask {np.mean(mask_true_positive_precent_out_of_mask)} with std of {np.std(mask_true_positive_precent_out_of_mask)} for {(val_i + 1)} clouds')
        print(f'LWC est score mask: mean relative error {np.mean(lwc_relative_err_est_mask_th)} with std of {np.std(lwc_relative_err_est_mask_th)} for {(val_i + 1)} clouds')
        print(f'LWC est score mask: mean relative mass error {np.mean(lwc_relative_mass_err_est_mask_th)} with std of {np.std(lwc_relative_mass_err_est_mask_th)} for {(val_i + 1)} clouds')
        print(f'Reff est score mask: mean relative error {np.mean(reff_relative_err_est_mask_th)} with std of {np.std(reff_relative_err_est_mask_th)} for {(val_i + 1)} clouds')
        print(f'Reff est score mask: mean relative mass error {np.mean(reff_relative_mass_err_est_mask_th)} with std of {np.std(reff_relative_mass_err_est_mask_th)} for {(val_i + 1)} clouds')
        print(f'Veff est score mask: mean relative error {np.mean(veff_relative_err_est_mask_th)} with std of {np.std(veff_relative_err_est_mask_th)} for {(val_i + 1)} clouds')
        print(f'Veff est score mask: mean relative mass error {np.mean(veff_relative_mass_err_est_mask_th)} with std of {np.std(veff_relative_mass_err_est_mask_th)} for {(val_i + 1)} clouds')
    # sio.savemat(f'numerical_results.mat', {'relative_err': relative_err, 'relative_mass_err': relative_mass_err,
    #                                        'l2_err': l2_err, 'batch_time_net': batch_time_net})

    # masked = relative_err < 2
    # relative_err1 = relative_err[masked]
    # relative_mass_err1 = relative_mass_err[masked]
    # l2_err1 = l2_err[masked]
    #
    # print(f'mean relative error w/o outliers {np.mean(relative_err1)} with std of {np.std(relative_err1)} for {relative_err1.shape[0]} clouds')
    # print(f'mean relative mass error w/o outliers {np.mean(relative_mass_err1)} with std of {np.std(relative_mass_err1)} for {relative_mass_err1.shape[0]} clouds')
    # print(f'mean L2 error w/o outliers {np.mean(l2_err1)} with std of {np.std(l2_err1)} for {l2_err1.shape[0]} clouds')

    if writer:
        writer._iter = iteration
        writer._dataset = 'val'#.format(val_i)
        writer.monitor_loss(loss_val)
        # writer.monitor_scatterer_error(relative_mass_err, relative_err)
        # writer.monitor_images(val_image)


if __name__ == "__main__":
    main()


