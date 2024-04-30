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
# sys.path.insert(0, '/home/roironen/pyshdom-NN/projects')
import hydra
import numpy as np
from VIPCT.visualization import SummaryWriter
from VIPCT.microphysics_dataset import get_cloud_microphysics_datasets, trivial_collate
from VIPCT.CTnet import *
from VIPCT.util.stats import Stats
from omegaconf import DictConfig
import torch
# from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup

relative_error = lambda ext_est, ext_gt, eps=1e-6 : torch.norm(ext_est.view(-1) - ext_gt.view(-1),p=1) / (torch.norm(ext_gt.view(-1),p=1) + eps)
mass_error = lambda ext_est, ext_gt, eps=1e-6 : (torch.norm(ext_gt.view(-1),p=1) - torch.norm(ext_est.view(-1),p=1)) / (torch.norm(ext_gt.view(-1),p=1) + eps)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
rho_water = 1e6  # g/m^3

@hydra.main(config_path=CONFIG_DIR, config_name="microphysics_train_w_env")
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

    # Load the training/validation data.
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # DATA_DIR = os.path.join(current_dir, "data")
    train_dataset, val_dataset = get_cloud_microphysics_datasets(
        cfg=cfg
    )

    # Initialize the CT model.
    model = CTnetMicrophysics(cfg=cfg, n_cam=cfg.data.n_cam)

    # Move the model to the relevant device.
    model.to(device)
    # Init stats to None before loading.
    stats = None
    optimizer_state_dict = None
    start_epoch = 0

    #
    log_dir = os.getcwd()
    writer = SummaryWriter(log_dir)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    checkpoint_resume_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_resume_path)
    if len(checkpoint_dir) > 0:
        # Make the root of the experiment directory.
        # checkpoint_dir = os.path.split(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume training if requested.
    if cfg.resume and os.path.isfile(checkpoint_resume_path):
        print(f"Resuming from checkpoint {checkpoint_resume_path}.")
        loaded_data = torch.load(checkpoint_resume_path, map_location=device)
        model.load_state_dict(loaded_data["model"])
        # stats = pickle.loads(loaded_data["stats"])
        # print(f"   => resuming from epoch {stats.epoch}.")
        # optimizer_state_dict = loaded_data["optimizer"]
        # start_epoch = stats.epoch

    init_lr = cfg.optimizer.lr
    if cfg.env_training:
        assert cfg.backbone.env_params_num, "env_training must include env params."
        print(f"Starting env training from checkpoint {checkpoint_resume_path}.")
        loaded_data = torch.load(checkpoint_resume_path, map_location=device)
        loaded_image_encoder = {k[15:]: v for k, v in loaded_data['model'].items() if k[:15]=='_image_encoder.'}
        curr_image_encoder_keys = model._image_encoder.state_dict().keys()
        image_encoder_keys_diff = list(curr_image_encoder_keys-loaded_image_encoder.keys())
        assert torch.all(torch.tensor([("insert_env" in key) for key in image_encoder_keys_diff], requires_grad=False))\
            , "problem with loading the image encoder"
        loaded_geometry_encoder = {k[8:]: v for k, v in loaded_data['model'].items() if k[:8] == 'mlp_xyz.'}
        loaded_camera_encoder = {k[15:]: v for k, v in loaded_data['model'].items() if k[:15] == 'mlp_cam_center.'}
        loaded_decoder = {k[8:]: v for k, v in loaded_data['model'].items() if k[:8] == 'decoder.'}
        model._image_encoder.load_state_dict(loaded_image_encoder, strict=False)
        model.mlp_xyz.load_state_dict(loaded_geometry_encoder, strict=True)
        model.mlp_cam_center.load_state_dict(loaded_camera_encoder, strict=True)
        model.decoder.load_state_dict(loaded_decoder, strict=True)
        init_lr = cfg.optimizer.lr/5

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=init_lr,
        weight_decay=cfg.optimizer.wd,
    )

    # Load the optimizer state dict in case we are resuming.
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        optimizer.last_epoch = start_epoch

    # Init the stats object.
    if stats is None:
        stats = Stats(
            ["loss", "loss_bce", "relative_error_lwc", "relative_error_reff", "relative_error_veff", "lr", "max_memory", "sec/it"],
        )

    # Learning rate scheduler setup.

    # Following the original code, we use exponential decay of the
    # learning rate: current_lr = base_lr * gamma ** (epoch / step_size)
    def lr_lambda(epoch):
        return cfg.optimizer.lr_scheduler_gamma ** (
            epoch #/ cfg.optimizer.lr_scheduler_step_size
        )

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )
    # lr_scheduler = create_lr_scheduler_with_warmup(
    #     lr_scheduler,
    #     warmup_start_value=cfg.optimizer.lr/100,
    #     warmup_duration=5000,
    #     warmup_end_value=cfg.optimizer.lr)

    # if cfg.data.precache_rays:
    #     # Precache the projection rays.
    #     model.eval()
    #     with torch.no_grad():
    #         for dataset in (train_dataset, val_dataset):
    #             for e in dataset:
    #                 cache_cameras = [cam.to(device) for cam in e["camera"]]
    #                 cache_camera_hashes = e["camera_idx"]#[e["camera_idx"] for e in dataset]
    #                 model.precache_rays(cache_cameras, cache_camera_hashes)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.optimizer.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=trivial_collate,
    )

    # The validation dataloader is just an endless stream of random samples.
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=trivial_collate,
        # sampler=torch.utils.data.RandomSampler(
        #     val_dataset,
        #     replacement=True,
        #     num_samples=cfg.optimizer.max_epochs,
        # ),
    )
    err = torch.nn.MSELoss()
    err_BCE = torch.nn.BCELoss()
    # err = torch.nn.L1Loss(reduction='sum')
    # Set the model to the training mode.
    model.train().float()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # Run the main training loop.
    iteration = -1
    if writer:
        val_scatter_ind = np.random.permutation(len(val_dataloader))[:5]
    for epoch in range(start_epoch, cfg.optimizer.max_epochs):
        for i, batch in enumerate(train_dataloader):
            iteration += 1
            # lr_scheduler(None)
            if iteration % (cfg.stats_print_interval) == 0 and iteration > 0:
                stats.new_epoch()  # Init a new epoch.
            if iteration in cfg.optimizer.iter_steps:
                # Adjust the learning rate.
                lr_scheduler.step()

            images, microphysics, grid, image_sizes, projection_matrix, camera_center, masks, env_params = batch#[0]#.values()

            images = torch.tensor(images, device=device).float()
            env_params = torch.tensor(env_params, device=device).float()
            volume = Volumes(torch.tensor(microphysics, device=device).float(), grid)
            cameras = PerspectiveCameras(image_size=image_sizes,P=torch.tensor(projection_matrix, device=device).float(),
                                         camera_center= torch.tensor(camera_center, device=device).float(), device=device)
            masks = [torch.tensor(mask) if mask is not None else mask for mask in masks]
            # if model.mask_type == 'gt_mask':
            #     masks = volume.extinctions > volume._ext_thr
            # R = torch.FloatTensor().to(device)
            # T = torch.FloatTensor().to(device)
            # for cam in camera:
            #     R = torch.cat((R, cam.R), dim=0)
            #     T = torch.cat((T, cam.T), dim=0)
            # camera = PerspectiveCameras(device=device, R=R, T=T)
            if torch.sum(torch.tensor([(mask).sum() if mask is not None else mask for mask in masks])) == 0:
                continue
            optimizer.zero_grad()

            # Run the forward pass of the model.
            out = model(
                cameras,
                images,
                volume,
                masks,
                env_params[:, :, :cfg.backbone.env_params_num]
            )

            if cfg.optimizer.lwc_loss == 'L2_relative_error_w_veff_w_score':
                est_lwc = [ext_est[:, 0] for ext_est in out["output"]]
                est_mask = [ext_est[:, 1] for ext_est in out["output"]]
                est_reff = [ext_est[:, 2] for ext_est in out["output"]]
                est_veff = [ext_est[:, 3] for ext_est in out["output"]]

                gt_lwc = [ext_gt[:, 0] for ext_gt in out["volume"]]
                gt_mask = [(ext_gt[:, 0]>0).float() for ext_gt in out["volume"]]
                gt_reff = [ext_gt[:, 1] for ext_gt in out["volume"]]
                gt_veff = [ext_gt[:, 2] for ext_gt in out["volume"]]
            else:
                est_lwc = [ext_est[:,0] for ext_est in out["output"]]
                est_reff = [ext_est[:, 1] for ext_est in out["output"]]
                est_veff = [ext_est[:, 2] for ext_est in out["output"]]

                gt_lwc = [ext_gt[:,0] for ext_gt in out["volume"]]
                gt_reff = [ext_gt[:, 1] for ext_gt in out["volume"]]
                gt_veff = [ext_gt[:, 2] for ext_gt in out["volume"]]

            gt_height = [torch.zeros_like(ext_gt[:,0]) for ext_gt in out["volume"]]
            for list_ind in torch.arange(len(gt_veff)):
                for ind in torch.arange(len(gt_veff[0])):
                    sample_inds = (torch.abs(volume.extinctions[:,0,:,:]-gt_lwc[0][ind]) +
                                        torch.abs(volume.extinctions[:,1,:,:]-gt_reff[0][ind]) +
                                        torch.abs(volume.extinctions[:,2,:,:]-gt_veff[0][ind]) < 1e-6).nonzero()[0,1:]
                    gt_height[list_ind][ind] = float(volume._grid[0][2][int(sample_inds[2])])

            # The loss is a sum of coarse and fine MSEs
            if cfg.optimizer.lwc_loss == 'L2_relative_error':
                # reff uses cross entropy and LWC l2 loss
                loss_lwc = [err(est.squeeze(),gt.squeeze())/(torch.norm(gt.squeeze())+ 1e-4) for est, gt in zip(est_lwc, gt_lwc)]
                loss_reff = [err(est.squeeze()[gt_lwc[i].squeeze()!=0],gt.squeeze()[gt_lwc[i].squeeze()!=0])/(torch.norm(gt.squeeze()[gt_lwc[i].squeeze()!=0])+ 1e-4) for i,(est, gt) in enumerate(zip(est_reff, gt_reff))]
                # loss_veff = [err(est.squeeze(),gt.squeeze())/(torch.norm(gt.squeeze())+ 1e-4) for est, gt in zip(est_veff, gt_veff)]
                # loss_index = torch.floor(torch.tensor(iteration % (cfg.optimizer.change_loss_interval*3))/cfg.optimizer.change_loss_interval)
                # if loss_index == 0:
                #     loss = torch.mean(torch.stack(loss_reff)) # * 100 + torch.mean(torch.stack(loss_reff)) /10 + torch.mean(torch.stack(loss_veff))
                # elif loss_index == 1:
                #     loss = torch.mean(torch.stack(loss_lwc))
                # else:
                #     loss = torch.mean(torch.stack(loss_veff))
                alpha_gt = torch.stack(gt_veff) ** (-1) - 2
                beta_gt = 1 / (torch.stack(gt_reff) * torch.stack(gt_veff))
                alpha_est = torch.stack(gt_veff) ** (-1) - 2 # torch.stack(est_veff) ** (-1) - 2
                beta_est = 1 / (torch.stack(est_reff) * torch.stack(gt_veff)) # 1 / (torch.stack(est_reff) * torch.stack(est_veff))
                # cross_entropy = alpha_gt * beta_est / beta_gt + torch.lgamma(alpha_est) - alpha_est * torch.log(beta_est) \
                #        - (alpha_est - 1) * torch.digamma(alpha_gt) + (alpha_est - 1) * torch.log(beta_gt)
                # cross_entropy[torch.stack(gt_lwc)==0] = 0
                # loss = torch.mean(cross_entropy)+100*torch.mean(torch.stack(loss_lwc))
                #norm_loss_reff = (1/(torch.stack(gt_veff)+1e-4))*torch.stack(loss_reff)
                #norm_loss_reff[torch.stack(gt_lwc)==0] = 0
                if iteration<25000:
                    loss = torch.mean(torch.stack(loss_lwc))
                else:
                    loss = torch.mean(torch.stack(loss_reff)) + torch.mean(torch.stack(loss_lwc))
                if loss.isnan():
                    aa=9

            if cfg.optimizer.lwc_loss == 'L2_relative_error_w_veff':
                loss_lwc = [err(est.squeeze(),gt.squeeze())/(torch.norm(gt.squeeze())+ 1e-4) for est, gt in zip(est_lwc, gt_lwc)]
                loss_reff = [err(est.squeeze()[gt_lwc[i].squeeze()!=0],gt.squeeze()[gt_lwc[i].squeeze()!=0])/(torch.norm(gt.squeeze()[gt_lwc[i].squeeze()!=0])+ 1e-4) for i,(est, gt) in enumerate(zip(est_reff, gt_reff))]
                loss_veff = [err(est.squeeze()[gt_lwc[i].squeeze()!=0],gt.squeeze()[gt_lwc[i].squeeze()!=0])/(torch.norm(gt.squeeze()[gt_lwc[i].squeeze()!=0])+ 1e-4) for i,(est, gt) in enumerate(zip(est_veff, gt_veff))]
                # loss_index = torch.floor(torch.tensor(iteration % (cfg.optimizer.change_loss_interval*3))/cfg.optimizer.change_loss_interval)
                # if loss_index == 0:
                #     loss = torch.mean(torch.stack(loss_reff)) # * 100 + torch.mean(torch.stack(loss_reff)) /10 + torch.mean(torch.stack(loss_veff))
                # elif loss_index == 1:
                #     loss = torch.mean(torch.stack(loss_lwc))
                # else:
                #     loss = torch.mean(torch.stack(loss_veff))
                alpha_gt = torch.stack(gt_veff) ** (-1) - 2
                beta_gt = 1 / (torch.stack(gt_reff) * torch.stack(gt_veff))
                alpha_est = torch.stack(gt_veff) ** (-1) - 2 # torch.stack(est_veff) ** (-1) - 2
                beta_est = 1 / (torch.stack(est_reff) * torch.stack(gt_veff)) # 1 / (torch.stack(est_reff) * torch.stack(est_veff))
                # cross_entropy = alpha_gt * beta_est / beta_gt + torch.lgamma(alpha_est) - alpha_est * torch.log(beta_est) \
                #        - (alpha_est - 1) * torch.digamma(alpha_gt) + (alpha_est - 1) * torch.log(beta_gt)
                # cross_entropy[torch.stack(gt_lwc)==0] = 0
                # loss = torch.mean(cross_entropy)+100*torch.mean(torch.stack(loss_lwc))
                #norm_loss_reff = (1/(torch.stack(gt_veff)+1e-4))*torch.stack(loss_reff)
                #norm_loss_reff[torch.stack(gt_lwc)==0] = 0
                loss = torch.mean(torch.stack(loss_reff)) + torch.mean(torch.stack(loss_lwc)) +torch.mean(torch.stack(loss_veff))
                loss_mask = (torch.tensor(0.0),torch.tensor(0.0))
                if loss.isnan():
                    aa=9

            if cfg.optimizer.lwc_loss == 'L2_relative_error_w_veff_w_score':
                loss_mask = [err_BCE(est.squeeze(),gt.squeeze()) for est, gt in zip(est_mask, gt_mask)]
                loss_lwc = [err(est.squeeze(),gt.squeeze())/(torch.norm(gt.squeeze())+ 1e-4) for est, gt in zip(est_lwc, gt_lwc)]
                loss_reff = [err(est.squeeze()[gt_lwc[i].squeeze()!=0],gt.squeeze()[gt_lwc[i].squeeze()!=0])/(torch.norm(gt.squeeze()[gt_lwc[i].squeeze()!=0])+ 1e-4) for i,(est, gt) in enumerate(zip(est_reff, gt_reff))]
                loss_veff = [err(est.squeeze()[gt_lwc[i].squeeze()!=0],gt.squeeze()[gt_lwc[i].squeeze()!=0])/(torch.norm(gt.squeeze()[gt_lwc[i].squeeze()!=0])+ 1e-4) for i,(est, gt) in enumerate(zip(est_veff, gt_veff))]
                # loss_index = torch.floor(torch.tensor(iteration % (cfg.optimizer.change_loss_interval*3))/cfg.optimizer.change_loss_interval)
                # if loss_index == 0:
                #     loss = torch.mean(torch.stack(loss_reff)) # * 100 + torch.mean(torch.stack(loss_reff)) /10 + torch.mean(torch.stack(loss_veff))
                # elif loss_index == 1:
                #     loss = torch.mean(torch.stack(loss_lwc))
                # else:
                #     loss = torch.mean(torch.stack(loss_veff))
                alpha_gt = torch.stack(gt_veff) ** (-1) - 2
                beta_gt = 1 / (torch.stack(gt_reff) * torch.stack(gt_veff))
                alpha_est = torch.stack(gt_veff) ** (-1) - 2 # torch.stack(est_veff) ** (-1) - 2
                beta_est = 1 / (torch.stack(est_reff) * torch.stack(gt_veff)) # 1 / (torch.stack(est_reff) * torch.stack(est_veff))
                # cross_entropy = alpha_gt * beta_est / beta_gt + torch.lgamma(alpha_est) - alpha_est * torch.log(beta_est) \
                #        - (alpha_est - 1) * torch.digamma(alpha_gt) + (alpha_est - 1) * torch.log(beta_gt)
                # cross_entropy[torch.stack(gt_lwc)==0] = 0
                # loss = torch.mean(cross_entropy)+100*torch.mean(torch.stack(loss_lwc))
                #norm_loss_reff = (1/(torch.stack(gt_veff)+1e-4))*torch.stack(loss_reff)
                #norm_loss_reff[torch.stack(gt_lwc)==0] = 0
                loss = torch.mean(torch.stack(loss_reff)) + torch.mean(torch.stack(loss_lwc)) +torch.mean(torch.stack(loss_veff)) + torch.mean(torch.stack(loss_mask))
                if loss.isnan():
                    aa=9

            elif cfg.optimizer.lwc_loss == 'cross_entropy':
                # combined cross entropy using reff and LWC
                alpha_gt = (torch.stack(gt_veff)+1e-4) ** (-1) - 2
                beta_gt = 1 / (torch.stack(gt_reff) * torch.stack(gt_veff)+1e-4)
                alpha_est = (torch.stack(gt_veff) ** (-1)+1e-4) - 2  # torch.stack(est_veff) ** (-1) - 2
                beta_est = 1 / (torch.stack(est_reff) * torch.stack(
                    gt_veff)+1e-4)  # 1 / (torch.stack(est_reff) * torch.stack(est_veff))
                norm_cross_entropy = alpha_gt * beta_est / beta_gt + torch.lgamma(alpha_est) - alpha_est * torch.log(
                    beta_est) - (alpha_est - 1) * torch.digamma(alpha_gt) + (alpha_est - 1) * torch.log(beta_gt)
                num_of_droplets_gt = torch.stack(gt_lwc)*beta_gt**3/\
                                     ((4/3)*torch.tensor(np.pi)*rho_water*alpha_gt*(alpha_gt+1)*(alpha_gt+2))
                num_of_droplets_est = torch.stack(gt_lwc) * beta_est ** 3 / \
                                     ((4 / 3) * torch.tensor(np.pi) * rho_water * alpha_est * (alpha_est + 1) * (alpha_est + 2))
                cross_entropy = -num_of_droplets_gt*torch.log(num_of_droplets_est)+num_of_droplets_gt*norm_cross_entropy
                loss = torch.mean(cross_entropy)
            elif cfg.optimizer.lwc_loss == 'kl_div':
                # combined kl divergence using reff and LWC
                alpha_gt = (torch.stack(gt_veff) + 1e-4) ** (-1) - 2
                beta_gt = 1 / (torch.stack(gt_reff) * torch.stack(gt_veff) + 1e-4)
                alpha_est = (torch.stack(gt_veff) + 1e-4) ** (-1)  - 2  # torch.stack(est_veff) ** (-1) - 2
                beta_est = 1 / (torch.stack(est_reff) * torch.stack(
                    gt_veff) + 1e-4)  # 1 / (torch.stack(est_reff) * torch.stack(est_veff))
                norm_kl_div = -alpha_est-torch.lgamma(alpha_est)\
                              +(alpha_est-1) * torch.digamma(alpha_est)+torch.log(beta_est)\
                              +(alpha_est * beta_gt / beta_est + torch.lgamma(alpha_gt) - alpha_gt * torch.log(beta_gt)
                              - (alpha_gt - 1) * torch.digamma(alpha_est) + (alpha_gt - 1) * torch.log(beta_est))
                num_of_droplets_gt = torch.stack(gt_lwc) * (beta_gt * 1e6) ** 3 / \
                                     ((4 / 3) * torch.tensor(np.pi) * rho_water * (alpha_gt+1e-5) * (alpha_gt + 1) * (
                                                 alpha_gt + 2))
                num_of_droplets_est = torch.stack(est_lwc) * (beta_est * 1e6) ** 3 / \
                                      ((4 / 3) * torch.tensor(np.pi) * rho_water * (alpha_est+1e-5) * (alpha_est + 1) * (
                                                  alpha_est + 2))
                kl_div = torch.log(num_of_droplets_est)-torch.log(num_of_droplets_gt)+norm_kl_div
                loss = torch.mean(kl_div)
                if loss.isnan():
                    aa=9
            elif cfg.optimizer.lwc_loss == 'pdfs_L2_error':
                loss_mask_lwc = [est[gt==0]**2 for est, gt in zip(est_lwc, gt_lwc)]
                lwc_mask = (torch.stack(gt_lwc)>0)
                alpha_gt = (torch.stack(gt_veff)[(lwc_mask) & (torch.stack(gt_veff)<0.5)] + 1e-4) ** (-1) - 2
                beta_gt = 1 / (torch.stack(gt_reff)[(lwc_mask) & (torch.stack(gt_veff)<0.5)] * torch.stack(gt_veff)[(lwc_mask) & (torch.stack(gt_veff)<0.5)] + 1e-4)
                alpha_est = (torch.stack(gt_veff)[(lwc_mask) & (torch.stack(gt_veff)<0.5)] + 1e-4) ** (-1) - 2  # torch.stack(est_veff) ** (-1) - 2
                beta_est = 1 / (torch.stack(est_reff)[(lwc_mask) & (torch.stack(gt_veff)<0.5)] * torch.stack(gt_veff)[(lwc_mask) & (torch.stack(gt_veff)<0.5)] + 1e-4)  # 1 / (torch.stack(est_reff) * torch.stack(est_veff))
                num_of_droplets_gt = torch.stack(gt_lwc)[(lwc_mask) & (torch.stack(gt_veff)<0.5)] * (beta_gt * 1e6) ** 3 / \
                                     ((4 / 3) * torch.tensor(np.pi) * rho_water * (alpha_gt + 1e-5) * (alpha_gt + 1) * (
                                             alpha_gt + 2))
                num_of_droplets_est = torch.stack(est_lwc)[(lwc_mask) & (torch.stack(gt_veff)<0.5)] * (beta_est * 1e6) ** 3 / \
                                      ((4 / 3) * torch.tensor(np.pi) * rho_water * (alpha_est + 1e-5) * (alpha_est + 1) * (
                                               alpha_est + 2))

                # h = 0.1
                # r = torch.arange(0, 50, h, device=device)
                # dist_gt = [n_gt * (b_gt ** a_gt) * ((r ** (a_gt - 1)) * (torch.exp(-b_gt * r)) / torch.exp(torch.lgamma(a_gt)))
                #            if n_gt!=0 else n_gt * r for a_gt, b_gt, n_gt in zip(alpha_gt.squeeze(), beta_gt.squeeze(), num_of_droplets_gt.squeeze())]
                # dist_est = [n_est * (b_est ** a_est) * (r ** (a_est - 1)) * (torch.exp(-b_est * r)) / torch.exp(torch.lgamma(a_est))
                #            for a_est, b_est, n_est in zip(alpha_est.squeeze(), beta_est.squeeze(), num_of_droplets_est.squeeze())]
                h = 0.1
                r = torch.arange(0.01, 50.01, h, device=device).repeat(alpha_gt.shape[0], 1).T
                #dist_gt = num_of_droplets_gt * (1 / torch.exp(torch.lgamma(alpha_gt))) * (beta_gt ** alpha_gt) * (r ** (alpha_gt - 1)) * (
                #    torch.exp(-beta_gt * r)) / torch.exp(torch.lgamma(alpha_gt))
                #dist_est = num_of_droplets_est * (1 / torch.exp(torch.lgamma(alpha_est))) * (beta_est ** alpha_est) * (r ** (alpha_est - 1)) * (
                #    torch.exp(-beta_est * r))
                log_dist_est = torch.log(num_of_droplets_est)+alpha_est*torch.log(beta_est)-torch.lgamma(alpha_est)+(alpha_est-1)*torch.log(r)-beta_est*r
                log_dist_gt = torch.log(num_of_droplets_gt) + alpha_gt * torch.log(beta_gt) - torch.lgamma(alpha_gt) + (
                            alpha_gt - 1) * torch.log(r) - beta_gt * r
                loss_dist = torch.mean(((log_dist_est-log_dist_gt)**2).sum(dim=0))/1e4
                loss = torch.mean(torch.stack(loss_mask_lwc))+loss_dist
                if loss.isnan():
                    aa=9
            else:
                NotImplementedError()

            # Take the training step.
            loss.backward()

            # import copy
            # A = copy.deepcopy(model)
            # optimizer.step()
            # for a, b in zip(model._image_encoder.model.conv1.weight, A._image_encoder.model.conv1.weight):
            #     print(torch.sum((a - b) ** 2))

            optimizer.step()
            with torch.no_grad():
                relative_err_lwc = [relative_error(ext_est=ext_est,ext_gt=ext_gt) for ext_est, ext_gt in zip(est_lwc, gt_lwc)]#torch.norm(out["output"] - out["volume"],p=1,dim=-1) / (torch.norm(out["volume"],p=1,dim=-1) + 1e-6)
                relative_err_lwc = torch.tensor(relative_err_lwc).mean()
                relative_err_reff = [relative_error(ext_est=ext_est[gt_lwc[i]!=0], ext_gt=ext_gt[gt_lwc[i]!=0]) for i, (ext_est, ext_gt) in enumerate(zip(est_reff, gt_reff))]  # torch.norm(out["output"] - out["volume"],p=1,dim=-1) / (torch.norm(out["volume"],p=1,dim=-1) + 1e-6)
                relative_err_reff = torch.tensor(relative_err_reff).mean()
                relative_err_veff = [relative_error(ext_est=ext_est, ext_gt=ext_gt) for ext_est, ext_gt in zip(est_veff, gt_veff)]  # torch.norm(out["output"] - out["volume"],p=1,dim=-1) / (torch.norm(out["volume"],p=1,dim=-1) + 1e-6)
                relative_err_veff = torch.tensor(relative_err_veff).mean()

                relative_mass_err_lwc = [mass_error(ext_est=ext_est,ext_gt=ext_gt) for ext_est, ext_gt in  zip(est_lwc, gt_lwc)]#(torch.norm(out["output"],p=1,dim=-1) - torch.norm(out["volume"],p=1,dim=-1)) / (torch.norm(out["volume"],p=1,dim=-1) + 1e-6)
                relative_mass_err_lwc = torch.tensor(relative_mass_err_lwc).mean()
                relative_mass_err_reff = [mass_error(ext_est=ext_est[gt_lwc[i]!=0], ext_gt=ext_gt[gt_lwc[i]!=0]) for i, (ext_est, ext_gt) in enumerate(zip(est_reff, gt_reff))]#(torch.norm(out["output"],p=1,dim=-1) - torch.norm(out["volume"],p=1,dim=-1)) / (torch.norm(out["volume"],p=1,dim=-1) + 1e-6)
                relative_mass_err_reff = torch.tensor(relative_mass_err_reff).mean()
                relative_mass_err_veff = [mass_error(ext_est=ext_est, ext_gt=ext_gt) for ext_est, ext_gt in zip(est_veff, gt_veff)]  # (torch.norm(out["output"],p=1,dim=-1) - torch.norm(out["volume"],p=1,dim=-1)) / (torch.norm(out["volume"],p=1,dim=-1) + 1e-6)
                relative_mass_err_veff = torch.tensor(relative_mass_err_veff).mean()

                # if iteration % cfg.stats_print_interval == 0 and iteration > 0:
                #     h = 0.001
                #     r = torch.arange(0, 20, h, device=device)
                #     rand_voxel_ind = torch.randint(0, alpha_gt.shape[0], (1,))
                #
                #     dist_gt = (beta_gt[0,rand_voxel_ind] ** alpha_gt[0,rand_voxel_ind]) * (r ** (alpha_gt[0,rand_voxel_ind] - 1)) * (
                #                   torch.exp(-beta_gt[0,rand_voxel_ind] * r)) / torch.exp(torch.lgamma(alpha_gt[0,rand_voxel_ind]))
                #     dist_est = (beta_est[0,rand_voxel_ind] ** alpha_est[0,rand_voxel_ind]) * (r ** (alpha_est[0,rand_voxel_ind] - 1)) * (
                #                    torch.exp(-beta_est[0,rand_voxel_ind] * r)) / torch.exp(torch.lgamma(alpha_est[0,rand_voxel_ind]))

            # Update stats with the current metrics.
            stats.update(
                {"loss": float(loss), "loss_bce": float(torch.mean(torch.stack(loss_mask))), "relative_error_lwc": float(relative_err_lwc), "relative_error_reff": float(relative_err_reff), "relative_error_veff": float(relative_err_veff), "lr":  lr_scheduler.get_last_lr()[0],#optimizer.param_groups[0]['lr'],#lr_scheduler.get_last_lr()[0]
                 "max_memory": float(round(torch.cuda.max_memory_allocated()/1e6))},
                stat_set="train",
            )

            if iteration % cfg.stats_print_interval == 0 and iteration > 0:
                stats.print(stat_set="train")
                if writer:
                    writer._iter = iteration
                    writer._dataset = 'train'
                    writer.monitor_loss(loss.item())
                    if cfg.optimizer.lwc_loss == 'L2_relative_error_w_veff_w_score':
                        writer.monitor_loss(torch.mean(torch.stack(loss_mask)).item())
                    writer.monitor_scatterer_error(relative_mass_err_lwc, relative_err_lwc, 'lwc')
                    writer.monitor_scatterer_error(relative_mass_err_reff, relative_err_reff, 'reff')
                    writer.monitor_scatterer_error(relative_mass_err_veff, relative_err_veff, 'veff')
                    # writer.monitor_distributions(dist_est, dist_gt, r)
                    for ind in range(len(out["output"])):
                        writer.monitor_scatter_plot(est_lwc[ind], gt_lwc[ind],ind=ind,name='lwc')
                        writer.monitor_scatter_plot(est_reff[ind][gt_lwc[ind]!=0], gt_reff[ind][gt_lwc[ind]!=0],ind=ind,name='reff')
                        writer.monitor_scatter_plot(est_veff[ind], gt_veff[ind], ind=ind, name='veff_reff',
                                                    colorbar_param = gt_reff[ind], colorbar_name = 'gt_reff')
                        writer.monitor_scatter_plot(est_veff[ind], gt_veff[ind], ind=ind, name='veff_height',
                                                    colorbar_param=gt_height[ind], colorbar_name='gt_height')
                    # writer.monitor_images(images)

            # Validation
            if iteration % cfg.validation_iter_interval == 0 and iteration > 0:
                loss_val = 0
                bce_err_mask = 0
                relative_err_lwc= 0
                relative_mass_err_lwc = 0
                relative_err_reff= 0
                relative_mass_err_reff = 0
                relative_err_veff = 0
                relative_mass_err_veff = 0

                val_i = 0
                for val_i, val_batch in enumerate(val_dataloader):

                # val_batch = next(val_dataloader.__iter__())

                    val_image, microphysics, grid, image_sizes, projection_matrix, camera_center, masks, val_env_params = val_batch#[0]#.values()
                    val_image = torch.tensor(val_image, device=device).float()
                    val_env_params = torch.tensor(val_env_params, device=device).float()
                    val_volume = Volumes(torch.tensor(microphysics, device=device).float(), grid)
                    val_camera = PerspectiveCameras(image_size=image_sizes,P=torch.tensor(projection_matrix, device=device).float(),
                                         camera_center= torch.tensor(camera_center, device=device).float(), device=device)
                    masks = [torch.tensor(mask) if mask is not None else mask for mask in masks]
                    # if model.val_mask_type == 'gt_mask':
                    #     masks = val_volume.extinctions > val_volume._ext_thr
                    if torch.sum(torch.tensor([(mask).sum() if mask is not None else mask for mask in masks])) == 0:
                        continue
                # Activate eval mode of the model (lets us do a full rendering pass).
                    model.eval()
                    with torch.no_grad():
                        val_out = model(
                            val_camera,
                            val_image,
                            val_volume,
                            masks,
                            val_env_params[:, :, :cfg.backbone.env_params_num]
                        )
                        if cfg.optimizer.lwc_loss == 'L2_relative_error_w_veff_w_score':
                            gt_shape = val_volume.extinctions.shape
                            est_vols = torch.zeros((gt_shape[0], gt_shape[1] + 1, gt_shape[2], gt_shape[3], gt_shape[4]),
                                                   device=val_volume.device)  # plus 1 for mask
                        else:
                            est_vols = torch.zeros(val_volume.extinctions.shape, device=val_volume.device)
                        if val_out['query_indices'] is None:
                            for i, (out_vol, m) in enumerate(zip(val_out["output"], masks)):
                                est_vols[i][m.squeeze(0)] = out_vol.squeeze(1)
                        else:
                            for est_vol, out_vol, m in zip(est_vols, val_out["output"], val_out['query_indices']):
                                est_vol = est_vol.reshape(est_vols.shape[1],-1)
                                est_vol[:,m] = out_vol.T  # .reshape(m.shape)[m]
                        assert len(est_vols)==1 ##TODO support validation with batch larger than 1
                        gt_vol = val_volume.extinctions[0].squeeze()
                        est_vols = est_vols.squeeze()

                        if cfg.optimizer.lwc_loss == 'L2_relative_error_w_veff_w_score':
                            est_lwc = est_vols[0][est_vols[0]!=0]
                            est_mask = est_vols[1][est_vols[0]!=0]
                            est_reff = est_vols[2][est_vols[0]!=0]
                            est_veff = est_vols[3][est_vols[0]!=0]

                            gt_lwc = gt_vol[0][est_vols[0]!=0]
                            gt_mask = ((gt_lwc>0).float())
                            gt_reff = gt_vol[1][est_vols[0]!=0]
                            gt_veff = gt_vol[2][est_vols[0]!=0]
                            gt_height = torch.tensor(val_volume._grid[0][2]).repeat(gt_vol[0].shape[0], gt_vol[0].shape[1], 1)
                            gt_height = gt_height[est_vols[0]!=0]
                        else:
                            est_lwc = est_vols[0]
                            est_reff = est_vols[1]
                            est_veff = est_vols[2]

                            gt_lwc = gt_vol[0]
                            gt_reff = gt_vol[1]
                            gt_veff = gt_vol[2]
                            gt_height = torch.tensor(val_volume._grid[0][2]).repeat(gt_lwc.shape[0], gt_lwc.shape[1], 1)
                        # loss_val += l1(val_out["output"], val_out["volume"]) / torch.sum(val_out["volume"]+1000)


                        est_lwc_for_loss = est_lwc.flatten()[est_lwc.flatten() > 0]
                        gt_lwc_for_loss = gt_lwc.flatten()[est_lwc.flatten() > 0]

                        gt_reff_for_loss = gt_reff.flatten()[est_lwc.flatten() > 0]
                        est_reff_for_loss = est_reff.flatten()[est_lwc.flatten() > 0]
                        est_veff_for_loss = est_veff.flatten()[est_lwc.flatten() > 0]
                        gt_veff_for_loss = gt_veff.flatten()[est_lwc.flatten() > 0]


                        if cfg.optimizer.lwc_loss == 'L2_relative_error':
                            loss_lwc = err(est_lwc_for_loss, gt_lwc_for_loss)
                            #loss_reff = err(est_reff_for_loss/(gt_veff_for_loss+1e-4), gt_reff_for_loss/(gt_veff_for_loss+1e-4))
                            loss_reff = err(est_reff_for_loss[gt_lwc_for_loss!=0], gt_reff_for_loss[gt_lwc_for_loss!=0])

                            loss_val += loss_reff + loss_lwc
                        elif cfg.optimizer.lwc_loss == 'L2_relative_error_w_veff':
                            loss_lwc = err(est_lwc_for_loss, gt_lwc_for_loss)
                            #loss_reff = err(est_reff_for_loss/(gt_veff_for_loss+1e-4), gt_reff_for_loss/(gt_veff_for_loss+1e-4))
                            loss_reff = err(est_reff_for_loss[gt_lwc_for_loss!=0], gt_reff_for_loss[gt_lwc_for_loss!=0])
                            loss_veff = err(est_veff_for_loss[gt_lwc_for_loss != 0], gt_veff_for_loss[gt_lwc_for_loss != 0])
                            loss_mask = 0.0
                            loss_val += loss_reff + loss_lwc + loss_veff
                        elif cfg.optimizer.lwc_loss == 'L2_relative_error_w_veff_w_score':
                            est_mask_for_loss = est_mask.flatten()[est_lwc.flatten() > 0]
                            gt_mask_for_loss = gt_mask.flatten()[est_lwc.flatten() > 0]
                            loss_lwc = err(est_lwc_for_loss, gt_lwc_for_loss)
                            loss_mask = err_BCE(est_mask_for_loss, gt_mask_for_loss)
                            loss_reff = err(est_reff_for_loss[gt_lwc_for_loss != 0],
                                            gt_reff_for_loss[gt_lwc_for_loss != 0])
                            loss_veff = err(est_veff_for_loss[gt_lwc_for_loss != 0],
                                            gt_veff_for_loss[gt_lwc_for_loss != 0])

                            loss_val += loss_reff + loss_lwc + loss_veff + loss_mask
                        elif cfg.optimizer.lwc_loss == 'pdfs_L2_error':
                            loss_mask_lwc = est_lwc_for_loss[gt_lwc_for_loss==0]**2
                            lwc_mask = (gt_lwc_for_loss > 0)
                            alpha_gt = (gt_veff_for_loss[(lwc_mask) & (gt_veff_for_loss<0.5)] + 1e-4) ** (-1) - 2
                            beta_gt = 1 / (gt_reff_for_loss[(lwc_mask) & (gt_veff_for_loss<0.5)] * gt_veff_for_loss[(lwc_mask) & (gt_veff_for_loss<0.5)] + 1e-4)
                            alpha_est = (gt_veff_for_loss[(lwc_mask) & (gt_veff_for_loss<0.5)] + 1e-4) ** (-1) - 2  # torch.stack(est_veff) ** (-1) - 2
                            beta_est = 1 / (est_reff_for_loss[(lwc_mask) & (gt_veff_for_loss<0.5)] * gt_veff_for_loss[(lwc_mask) & (gt_veff_for_loss<0.5)] + 1e-4)
                            num_of_droplets_gt = gt_lwc_for_loss[(lwc_mask) & (gt_veff_for_loss<0.5)] * (beta_gt * 1e6) ** 3 / \
                                     ((4 / 3) * torch.tensor(np.pi) * rho_water * (alpha_gt + 1e-5) * (alpha_gt + 1) * (
                                             alpha_gt + 2))
                            num_of_droplets_est = est_lwc_for_loss[(lwc_mask) & (gt_veff_for_loss<0.5)] * (beta_est * 1e6) ** 3 / \
                                      ((4 / 3) * torch.tensor(np.pi) * rho_water * (alpha_est + 1e-5) * (alpha_est + 1) * (
                                               alpha_est + 2))
                            h = 0.1
                            r = torch.arange(0.01, 50.01, h, device=device).repeat(alpha_gt.shape[0], 1).T
                            log_dist_est = torch.log(num_of_droplets_est)+alpha_est*torch.log(beta_est)-torch.lgamma(alpha_est)+(alpha_est-1)*torch.log(r)-beta_est*r
                            log_dist_gt = torch.log(num_of_droplets_gt) + alpha_gt * torch.log(beta_gt) - torch.lgamma(alpha_gt) + (
                                alpha_gt - 1) * torch.log(r) - beta_gt * r
                            loss_dist = torch.mean(((log_dist_est-log_dist_gt)**2).sum(dim=0))/1e4
                            loss_val += torch.mean(loss_mask_lwc)+loss_dist

                        relative_err_lwc += relative_error(ext_est=est_lwc,ext_gt=gt_lwc).item()
                        relative_mass_err_lwc += mass_error(ext_est=est_lwc,ext_gt=gt_lwc).item()

                        relative_err_reff += relative_error(ext_est=est_reff[gt_lwc!=0],ext_gt=gt_reff[gt_lwc!=0]).item()
                        relative_mass_err_reff += mass_error(ext_est=est_reff[gt_lwc!=0],ext_gt=gt_reff[gt_lwc!=0]).item()

                        relative_err_veff += relative_error(ext_est=est_veff, ext_gt=gt_veff).item()
                        relative_mass_err_veff += mass_error(ext_est=est_veff, ext_gt=gt_veff).item()

                        bce_err_mask += loss_mask

                        if writer:
                            writer._iter = iteration
                            writer._dataset = 'val'  # .format(val_i)
                            if val_i in val_scatter_ind:
                                writer.monitor_scatter_plot(est_lwc, gt_lwc,ind=val_i,dilute_percent=1,name='lwc')
                                writer.monitor_scatter_plot(est_reff[gt_lwc!=0], gt_reff[gt_lwc!=0],ind=val_i,dilute_percent=1,name='reff')
                                writer.monitor_scatter_plot(est_veff, gt_veff, ind=val_i,dilute_percent=1, name='veff_reff',
                                                        colorbar_param=gt_reff, colorbar_name='gt_reff')
                                writer.monitor_scatter_plot(est_veff, gt_veff, ind=val_i,dilute_percent=1, name='veff_height',
                                                        colorbar_param=gt_height, colorbar_name='gt_height')


                loss_val /= (val_i + 1)
                relative_err_lwc /= (val_i + 1)
                relative_mass_err_lwc /= (val_i+1)

                relative_err_reff /= (val_i + 1)
                relative_mass_err_reff /= (val_i+1)

                relative_err_veff /= (val_i + 1)
                relative_mass_err_veff /= (val_i + 1)

                bce_err_mask /= (val_i + 1)

                # Update stats with the validation metrics.
                stats.update({"loss": float(loss_val), "loss_bce": float(bce_err_mask), "relative_error_lwc": float(relative_err_lwc),
                              "relative_error_reff": float(relative_err_reff),
                              "relative_error_veff": float(relative_err_veff),}, stat_set="val")

                if writer:
                    writer._iter = iteration
                    writer._dataset = 'val'#.format(val_i)
                    writer.monitor_loss(loss_val)
                    writer.monitor_loss(bce_err_mask)
                    writer.monitor_scatterer_error(relative_mass_err_lwc, relative_err_lwc, 'lwc')
                    writer.monitor_scatterer_error(relative_mass_err_reff, relative_err_reff, 'reff')
                    writer.monitor_scatterer_error(relative_mass_err_veff, relative_err_veff, 'veff')
                    # writer.monitor_images(val_image)

                stats.print(stat_set="val")



                # Set the model back to train mode.
                model.train()

                # Checkpoint.
            if (
                iteration % cfg.checkpoint_iteration_interval == 0
                and len(checkpoint_dir) > 0
                and iteration > 0
            ):
                curr_checkpoint_path = os.path.join(checkpoint_dir,f'cp_{iteration}.pth')
                print(f"Storing checkpoint {curr_checkpoint_path}.")
                data_to_store = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "stats": pickle.dumps(stats),
                }
                torch.save(data_to_store, curr_checkpoint_path)


if __name__ == "__main__":
    main()
