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
    train_dataset, val_dataset, _ = get_cloud_microphysics_datasets(cfg=cfg)

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

    assert (cfg.env_training_image_encoder + cfg.env_training_separate_encoder + cfg.env_training_no_input <= 1)
    if cfg.env_training_image_encoder:
        assert cfg.data.env_params_num, "env_training must include env params."
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
        if ('varsat' not in cfg.data.dataset_name) and ('varenv' not in cfg.data.dataset_name):
            for param in model.mlp_xyz.parameters():
                param.requires_grad = False
            for param in model.mlp_cam_center.parameters():
                param.requires_grad = False

        init_lr = cfg.optimizer.lr/5
    elif cfg.env_training_separate_encoder:
        assert cfg.data.env_params_num, "env_training must include env params."
        # print(f"Starting env training from checkpoint {checkpoint_resume_path}.")
        # loaded_data = torch.load(checkpoint_resume_path, map_location=device)
        # loaded_image_encoder = {k[15:]: v for k, v in loaded_data['model'].items() if k[:15]=='_image_encoder.'}
        # loaded_geometry_encoder = {k[8:]: v for k, v in loaded_data['model'].items() if k[:8] == 'mlp_xyz.'}
        # loaded_camera_encoder = {k[15:]: v for k, v in loaded_data['model'].items() if k[:15] == 'mlp_cam_center.'}
        # model._image_encoder.load_state_dict(loaded_image_encoder, strict=True)
        # model.mlp_xyz.load_state_dict(loaded_geometry_encoder, strict=True)
        # model.mlp_cam_center.load_state_dict(loaded_camera_encoder, strict=True)
        # # freeze all image encoder but the last layer:
        # for param in model._image_encoder.parameters():
        #     param.requires_grad = False
        # unfreeze_last_layer_code = 'for param in model._image_encoder.model.layer' + str(cfg.backbone.num_layers) + '.parameters(): param.requires_grad = True'
        # exec(unfreeze_last_layer_code)
        # if ('varsat' not in cfg.data.dataset_name) and ('varenv' not in cfg.data.dataset_name):
        #     for param in model.mlp_xyz.parameters():
        #         param.requires_grad = False
        #     for param in model.mlp_cam_center.parameters():
        #         param.requires_grad = False
        #
        # init_lr = cfg.optimizer.lr/5
    elif cfg.env_training_no_input:
        print(f"Starting env training from checkpoint {checkpoint_resume_path}.")
        loaded_data = torch.load(checkpoint_resume_path, map_location=device)
        model.load_state_dict(loaded_data["model"], strict=True)
        init_lr = cfg.optimizer.lr / 5


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
        if "score" in cfg.optimizer.loss:
            stats = Stats(
                ["loss", "loss_bce", "relative_error_lwc", "relative_error_reff", "relative_error_veff", "lr", "max_memory", "sec/it"],
            )
        else:
            stats = Stats(
                ["loss", "relative_error_lwc", "relative_error_reff", "relative_error_veff", "lr",
                 "max_memory", "sec/it"],
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
    )
    err = torch.nn.MSELoss()
    err_BCE = torch.nn.BCELoss()

    # Set the model to the training mode.
    model.train().float()

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
            if cfg.data.env_params_num == 0:
                env_params = None
            if env_params is not None:
                env_params = torch.tensor(env_params, device=device).float()
                if cfg.data.env_params_num == 2:  # sun az and zen
                    env_params = env_params[:, :, -2:]
                else:
                    env_params = env_params[:, :, :cfg.data.env_params_num]
            volume = Volumes(torch.tensor(microphysics, device=device).float(), grid)
            cameras = PerspectiveCameras(image_size=image_sizes,P=torch.tensor(projection_matrix, device=device).float(),
                                         camera_center= torch.tensor(camera_center, device=device).float(), device=device)
            masks = [torch.tensor(mask) if mask is not None else mask for mask in masks]
            if torch.sum(torch.tensor([(mask).sum() if mask is not None else mask for mask in masks])) == 0:
                continue
            optimizer.zero_grad()

            #  Run the forward pass of the model.
            out = model(
                cameras,
                images,
                volume,
                masks,
                env_params
            )

            if 'score' in cfg.optimizer.loss:
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

            if cfg.optimizer.loss == 'L2_relative_error_w_veff':
                loss_lwc = [err(est.squeeze(),gt.squeeze())/(torch.norm(gt.squeeze())+ 1e-4) for est, gt in zip(est_lwc, gt_lwc)]
                loss_reff = [err(est.squeeze()[gt_lwc[i].squeeze()!=0],gt.squeeze()[gt_lwc[i].squeeze()!=0])/(torch.norm(gt.squeeze()[gt_lwc[i].squeeze()!=0])+ 1e-4) for i,(est, gt) in enumerate(zip(est_reff, gt_reff))]
                loss_veff = [err(est.squeeze()[gt_lwc[i].squeeze()!=0],gt.squeeze()[gt_lwc[i].squeeze()!=0])/(torch.norm(gt.squeeze()[gt_lwc[i].squeeze()!=0])+ 1e-4) for i,(est, gt) in enumerate(zip(est_veff, gt_veff))]
                loss = torch.mean(torch.stack(loss_reff)) + 100*torch.mean(torch.stack(loss_lwc)) + 10*torch.mean(torch.stack(loss_veff))
                loss_mask = (torch.tensor(0.0),torch.tensor(0.0))
                if loss.isnan():
                    aa=9
            elif cfg.optimizer.loss == 'L2_relative_error_w_veff_w_score':
                loss_mask = [err_BCE(est.squeeze(),gt.squeeze()) for est, gt in zip(est_mask, gt_mask)]
                loss_lwc = [err(est.squeeze(),gt.squeeze())/(torch.norm(gt.squeeze())+ 1e-4) for est, gt in zip(est_lwc, gt_lwc)]
                loss_reff = [err(est.squeeze()[gt_lwc[i].squeeze()!=0],gt.squeeze()[gt_lwc[i].squeeze()!=0])/(torch.norm(gt.squeeze()[gt_lwc[i].squeeze()!=0])+ 1e-4) for i,(est, gt) in enumerate(zip(est_reff, gt_reff))]
                loss_veff = [err(est.squeeze()[gt_lwc[i].squeeze()!=0],gt.squeeze()[gt_lwc[i].squeeze()!=0])/(torch.norm(gt.squeeze()[gt_lwc[i].squeeze()!=0])+ 1e-4) for i,(est, gt) in enumerate(zip(est_veff, gt_veff))]
                loss = 0.1*torch.mean(torch.stack(loss_reff)) + torch.mean(torch.stack(loss_lwc)) + torch.mean(torch.stack(loss_veff)) + torch.mean(torch.stack(loss_mask))
                if loss.isnan():
                    aa=9
            elif cfg.optimizer.loss == 'L2_unitless_error_w_veff_w_score':
                loss_mask = [err_BCE(est.squeeze(),gt.squeeze()) for est, gt in zip(est_mask, gt_mask)]
                loss_lwc = [err(est.squeeze(),gt.squeeze())/(torch.norm(gt.squeeze())+ 1e-4)**2 for est, gt in zip(est_lwc, gt_lwc)]
                loss_reff = [err(est.squeeze()[gt_lwc[i].squeeze()!=0],gt.squeeze()[gt_lwc[i].squeeze()!=0])/(torch.norm(gt.squeeze()[gt_lwc[i].squeeze()!=0])+ 1e-4)**2 for i,(est, gt) in enumerate(zip(est_reff, gt_reff))]
                loss_veff = [err(est.squeeze()[gt_lwc[i].squeeze()!=0],gt.squeeze()[gt_lwc[i].squeeze()!=0])/(torch.norm(gt.squeeze()[gt_lwc[i].squeeze()!=0])+ 1e-4)**2 for i,(est, gt) in enumerate(zip(est_veff, gt_veff))]
                loss = torch.mean(torch.stack(loss_reff)) + torch.mean(torch.stack(loss_lwc)) + torch.mean(torch.stack(loss_veff)) + torch.mean(torch.stack(loss_mask))
                if loss.isnan():
                    aa=9
            elif cfg.optimizer.loss == 'L2_relative_error_beta_w_score':
                loss_mask = [err_BCE(est.squeeze(), gt.squeeze()) for est, gt in zip(est_mask, gt_mask)]
                # loss_lwc = [err(est.squeeze(), gt.squeeze()) / (torch.norm(gt.squeeze()) + 1e-4) for est, gt in
                #             zip(est_lwc, gt_lwc)]
                # loss_reff = [err(est.squeeze()[gt_lwc[i].squeeze() != 0], gt.squeeze()[gt_lwc[i].squeeze() != 0]) / (
                #             torch.norm(gt.squeeze()[gt_lwc[i].squeeze() != 0]) + 1e-4) for i, (est, gt) in
                #              enumerate(zip(est_reff, gt_reff))]
                loss_veff = [err(est.squeeze()[gt_lwc[i].squeeze() != 0], gt.squeeze()[gt_lwc[i].squeeze() != 0]) / (
                            torch.norm(gt.squeeze()[gt_lwc[i].squeeze() != 0]) + 1e-4) for i, (est, gt) in
                             enumerate(zip(est_veff, gt_veff))]
                gt_reff_tmp = []
                gt_beta = [(lwc / reff) for lwc, reff in zip(gt_lwc, gt_reff)]
                for beta in gt_beta:
                    beta[torch.isnan(beta)] = 0.
                est_beta = [(lwc / reff) for lwc, reff in zip(est_lwc, est_reff)]
                for beta in est_beta:
                    beta[torch.isnan(beta)] = 0.
                loss_beta = [err(est.squeeze(), gt.squeeze()) / (torch.norm(gt.squeeze()) + 1e-4)
                             for (est, gt) in zip(est_beta, gt_beta)]

                loss = torch.mean(torch.stack(loss_beta)) + torch.mean(torch.stack(loss_veff)) + torch.mean(torch.stack(loss_mask))
                if loss.isnan():
                    aa = 9
            elif cfg.optimizer.loss == 'L2_relative_error_just_lwc':
                loss_lwc = [err(est.squeeze(), gt.squeeze()) / (torch.norm(gt.squeeze()) + 1e-4) for est, gt in
                            zip(est_lwc, gt_lwc)]
                loss_reff = 0
                loss_veff = 0
                loss = torch.mean(torch.stack(loss_lwc))
                loss_mask = (torch.tensor(0.0), torch.tensor(0.0))
                if loss.isnan():
                    aa = 9
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

            # Update stats with the current metrics.
            if "score" in cfg.optimizer.loss:
                stats.update(
                    {"loss": float(loss), "loss_bce": float(torch.mean(torch.stack(loss_mask))), "relative_error_lwc": float(relative_err_lwc), "relative_error_reff": float(relative_err_reff), "relative_error_veff": float(relative_err_veff), "lr":  lr_scheduler.get_last_lr()[0],
                     "max_memory": float(round(torch.cuda.max_memory_allocated()/1e6))},
                    stat_set="train",
                )
            else:
                stats.update(
                    {"loss": float(loss),
                     "relative_error_lwc": float(relative_err_lwc), "relative_error_reff": float(relative_err_reff),
                     "relative_error_veff": float(relative_err_veff), "lr": lr_scheduler.get_last_lr()[0],
                     "max_memory": float(round(torch.cuda.max_memory_allocated() / 1e6))},
                    stat_set="train",
                )

            if iteration % cfg.stats_print_interval == 0 and iteration > 0:
                stats.print(stat_set="train")
                if writer:
                    writer._iter = iteration
                    writer._dataset = 'train'
                    writer.monitor_loss(loss.item())
                    if "score" in cfg.optimizer.loss:
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

                    val_image, microphysics, grid, image_sizes, projection_matrix, camera_center, masks, val_env_params = val_batch
                    val_image = torch.tensor(val_image, device=device).float()
                    if cfg.data.env_params_num == 0:
                        val_env_params = None
                    if val_env_params is not None:
                        val_env_params = torch.tensor(val_env_params, device=device).float()
                        if cfg.data.env_params_num == 2:  # sun az and zen
                            val_env_params = val_env_params[:, :, -2:]
                        else:
                            val_env_params = val_env_params[:, :, :cfg.data.env_params_num]
                    val_volume = Volumes(torch.tensor(microphysics, device=device).float(), grid)
                    val_camera = PerspectiveCameras(image_size=image_sizes,P=torch.tensor(projection_matrix, device=device).float(),
                                         camera_center= torch.tensor(camera_center, device=device).float(), device=device)
                    masks = [torch.tensor(mask) if mask is not None else mask for mask in masks]

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
                            val_env_params
                        )
                        if "score" in cfg.optimizer.loss:
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

                        if "score" in cfg.optimizer.loss:
                            est_lwc = est_vols[0][est_vols[0]!=0]
                            est_mask = est_vols[1][est_vols[0]!=0]
                            est_reff = est_vols[2][est_vols[0]!=0]
                            est_veff = est_vols[3][est_vols[0]!=0]

                            gt_lwc = gt_vol[0][est_vols[0]!=0]
                            gt_mask = ((gt_lwc>0).float())
                            gt_reff = gt_vol[1][est_vols[0]!=0]
                            gt_veff = gt_vol[2][est_vols[0]!=0]
                            gt_height = torch.tensor(np.array(list(val_volume._grid[0][2]), dtype=np.float32)).repeat(gt_vol[0].shape[0], gt_vol[0].shape[1], 1)
                            gt_height = gt_height[est_vols[0]!=0]
                        else:
                            est_lwc = est_vols[0]
                            est_reff = est_vols[1]
                            est_veff = est_vols[2]

                            gt_lwc = gt_vol[0]
                            gt_reff = gt_vol[1]
                            gt_veff = gt_vol[2]
                            gt_height = torch.tensor(np.array(list(val_volume._grid[0][2]), dtype=np.float32)).repeat(gt_lwc.shape[0], gt_lwc.shape[1], 1)


                        est_lwc_for_loss = est_lwc.flatten()[est_lwc.flatten() > 0]
                        gt_lwc_for_loss = gt_lwc.flatten()[est_lwc.flatten() > 0]
                        est_reff_for_loss = est_reff.flatten()[est_lwc.flatten() > 0]
                        gt_reff_for_loss = gt_reff.flatten()[est_lwc.flatten() > 0]
                        est_veff_for_loss = est_veff.flatten()[est_lwc.flatten() > 0]
                        gt_veff_for_loss = gt_veff.flatten()[est_lwc.flatten() > 0]

                        if cfg.optimizer.loss == 'L2_relative_error_w_veff':
                            loss_lwc = err(est_lwc_for_loss, gt_lwc_for_loss)
                            #loss_reff = err(est_reff_for_loss/(gt_veff_for_loss+1e-4), gt_reff_for_loss/(gt_veff_for_loss+1e-4))
                            loss_reff = err(est_reff_for_loss[gt_lwc_for_loss!=0], gt_reff_for_loss[gt_lwc_for_loss!=0])
                            loss_veff = err(est_veff_for_loss[gt_lwc_for_loss != 0], gt_veff_for_loss[gt_lwc_for_loss != 0])
                            loss_mask = 0.0
                            loss_val += loss_reff + loss_lwc + loss_veff
                        elif cfg.optimizer.loss == 'L2_relative_error_w_veff_w_score':
                            est_mask_for_loss = est_mask.flatten()[est_lwc.flatten() > 0]
                            gt_mask_for_loss = gt_mask.flatten()[est_lwc.flatten() > 0]
                            loss_lwc = err(est_lwc_for_loss, gt_lwc_for_loss)/(torch.norm(gt_lwc_for_loss)+ 1e-4)
                            loss_mask = err_BCE(est_mask_for_loss, gt_mask_for_loss)
                            loss_reff = err(est_reff_for_loss[gt_lwc_for_loss != 0],
                                            gt_reff_for_loss[gt_lwc_for_loss != 0])/(torch.norm(gt_reff_for_loss[gt_lwc_for_loss != 0])+ 1e-4)
                            loss_veff = err(est_veff_for_loss[gt_lwc_for_loss != 0],
                                            gt_veff_for_loss[gt_lwc_for_loss != 0])/(torch.norm(gt_veff_for_loss[gt_lwc_for_loss != 0])+ 1e-4)

                            loss_val += 0.1*loss_reff + loss_lwc + loss_veff + loss_mask
                        elif cfg.optimizer.loss == 'L2_unitless_error_w_veff_w_score':
                            est_mask_for_loss = est_mask.flatten()[est_lwc.flatten() > 0]
                            gt_mask_for_loss = gt_mask.flatten()[est_lwc.flatten() > 0]
                            loss_lwc = err(est_lwc_for_loss, gt_lwc_for_loss) / (torch.norm(gt_lwc_for_loss) + 1e-4)**2
                            loss_mask = err_BCE(est_mask_for_loss, gt_mask_for_loss)
                            loss_reff = err(est_reff_for_loss[gt_lwc_for_loss != 0],
                                            gt_reff_for_loss[gt_lwc_for_loss != 0]) / (
                                                    torch.norm(gt_reff_for_loss[gt_lwc_for_loss != 0]) + 1e-4)**2
                            loss_veff = err(est_veff_for_loss[gt_lwc_for_loss != 0],
                                            gt_veff_for_loss[gt_lwc_for_loss != 0]) / (
                                                    torch.norm(gt_veff_for_loss[gt_lwc_for_loss != 0]) + 1e-4)**2

                            loss_val += loss_reff + loss_lwc + loss_veff + loss_mask
                        elif cfg.optimizer.loss == 'L2_relative_error_just_lwc':
                            loss_lwc = err(est_lwc_for_loss, gt_lwc_for_loss)
                            loss_reff = 0.
                            loss_veff = 0.
                            loss_mask = 0.0
                            loss_val += loss_lwc

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
                                writer.monitor_scatter_plot(est_lwc, gt_lwc, ind=val_i, dilute_percent=1, name='lwc')
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
