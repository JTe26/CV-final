#!/usr/bin/env python3
import os
import sys
import json
import random
import datetime
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from opt import config_parser
from renderer import *
from utils import *
from dataLoader import dataset_dict

# Ensure reproducibility
torch.set_default_dtype(torch.float32)
torch.manual_seed(20211202)
np.random.seed(20211202)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr + self.batch]


@torch.no_grad()
def export_mesh(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha, _ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(
        alpha.cpu(),
        f'{args.ckpt[:-3]}.ply',
        bbox=tensorf.aabb.cpu(),
        level=0.005
    )


@torch.no_grad()
def render_test(args):
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(
        args.datadir, split='test',
        downsample=args.downsample_train,
        is_stack=True
    )
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    # render train set if requested
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(
            args.datadir, split='train',
            downsample=args.downsample_train,
            is_stack=True
        )
        PSNRs_train = evaluation(
            train_dataset, tensorf, args, renderer,
            f'{logfolder}/imgs_train_all/',
            N_vis=-1, N_samples=-1,
            white_bg=white_bg, ndc_ray=ndc_ray,
            device=device
        )
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_train)} <========================')

    # render test set
    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(
            test_dataset, tensorf, args, renderer,
            f'{logfolder}/{args.expname}/imgs_test_all/',
            N_vis=-1, N_samples=-1,
            white_bg=white_bg, ndc_ray=ndc_ray,
            device=device
        )

    # render along path if requested
    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(
            test_dataset, tensorf, c2ws, renderer,
            f'{logfolder}/{args.expname}/imgs_path_all/',
            N_vis=-1, N_samples=-1,
            white_bg=white_bg, ndc_ray=ndc_ray,
            device=device
        )


def reconstruction(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(
        args.datadir, split='train',
        downsample=args.downsample_train,
        is_stack=False
    )
    test_dataset = dataset(
        args.datadir, split='test',
        downsample=args.downsample_train,
        is_stack=True
    )
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # setup logging directory and TensorBoard writer
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)

    summary_writer = SummaryWriter(logfolder)
    print(f"Logging to {logfolder}")

    # init parameters
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    # build or load model
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(
            aabb, reso_cur, device,
            density_n_comp=args.n_lamb_sigma,
            appearance_n_comp=args.n_lamb_sh,
            app_dim=args.data_dim_color,
            near_far=near_far,
            shadingMode=args.shadingMode,
            alphaMask_thres=args.alpha_mask_thre,
            density_shift=args.density_shift,
            distance_scale=args.distance_scale,
            pos_pe=args.pos_pe,
            view_pe=args.view_pe,
            fea_pe=args.fea_pe,
            featureC=args.featureC,
            step_ratio=args.step_ratio,
            fea2denseAct=args.fea2denseAct
        )

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # prepare upsample and alpha‐mask schedules
    if args.upsamp_list is None:
        upsamp_list = []
    else:
        upsamp_list = args.upsamp_list
    if args.update_AlphaMask_list is None:
        update_AlphaMask_list = []
    else:
        update_AlphaMask_list = args.update_AlphaMask_list

    # compute voxel counts for each upsampling step
    N_voxel_list = (
        torch.round(
            torch.exp(
                torch.linspace(
                    np.log(args.N_voxel_init),
                    np.log(args.N_voxel_final),
                    len(upsamp_list) + 1
                )
            )
        ).long()
    ).tolist()[1:]

    # prepare ray sampling
    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    # weights for regularizers
    Ortho_reg_weight = args.Ortho_weight
    L1_reg_weight = args.L1_weight_inital
    TV_weight_density = args.TV_weight_density
    TV_weight_app = args.TV_weight_app
    tvreg = TVLoss()

    PSNRs, PSNRs_test = [], [0]

    pbar = tqdm(range(args.n_iters), file=sys.stdout, miniters=args.progress_refresh_rate)
    for iteration in pbar:
        # sample rays
        ray_idx = trainingSampler.nextids()
        rays_train = allrays[ray_idx]
        rgb_train = allrgbs[ray_idx].to(device)

        # forward pass
        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(
            rays_train, tensorf,
            chunk=args.batch_size,
            N_samples=nSamples,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
            is_train=True
        )

        # compute loss
        mse_loss = torch.mean((rgb_map - rgb_train) ** 2)
        total_loss = mse_loss
        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss = total_loss + Ortho_reg_weight * loss_reg
            summary_writer.add_scalar('train/reg_ortho', loss_reg.item(), iteration)
        if L1_reg_weight > 0:
            loss_l1 = tensorf.density_L1()
            total_loss = total_loss + L1_reg_weight * loss_l1
            summary_writer.add_scalar('train/reg_l1', loss_l1.item(), iteration)
        if TV_weight_density > 0:
            loss_tv_d = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv_d
            summary_writer.add_scalar('train/reg_tv_density', loss_tv_d.item(), iteration)
        if TV_weight_app > 0:
            loss_tv_a = tensorf.TV_loss_app(tvreg) * TV_weight_app
            total_loss = total_loss + loss_tv_a
            summary_writer.add_scalar('train/reg_tv_app', loss_tv_a.item(), iteration)

        # backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # log train metrics
        mse_scalar = mse_loss.item()
        PSNR = -10.0 * np.log(mse_scalar) / np.log(10.0)
        PSNRs.append(PSNR)
        summary_writer.add_scalar('train/mse', mse_scalar, iteration)
        summary_writer.add_scalar('train/psnr', PSNR, iteration)
        summary_writer.add_scalar('train/total_loss', total_loss.item(), iteration)

        # update tqdm
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f"Iter {iteration:05d}: train_psnr={np.mean(PSNRs):.2f}"
                f" train_mse={mse_scalar:.6f}"
            )
            PSNRs = []

        # periodic test‐set evaluation for visualization
        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis != 0:
            PSNRs_test = evaluation(
                test_dataset, tensorf, args, renderer,
                f'{logfolder}/imgs_vis/',
                N_vis=args.N_vis,
                prtx=f'{iteration:06d}_',
                N_samples=nSamples,
                white_bg=white_bg,
                ndc_ray=ndc_ray,
                device=device
            )
            mean_psnr_test = np.mean(PSNRs_test)
            summary_writer.add_scalar('test/psnr', mean_psnr_test, iteration)
            # convert back to mse for logging
            mean_mse_test = np.mean(10.0 ** (-np.array(PSNRs_test) / 10.0))
            summary_writer.add_scalar('test/mse', mean_mse_test, iteration)

        # alpha‐mask update & upsampling (omitted for brevity; unchanged)...
        # ...

    # save final model
    tensorf.save(f'{logfolder}/{args.expname}.th')

    # final full‐set test evaluation
    if args.render_test:
        PSNRs_test = evaluation(
            test_dataset, tensorf, args, renderer,
            f'{logfolder}/imgs_test_all/',
            N_vis=-1, N_samples=-1,
            white_bg=white_bg, ndc_ray=ndc_ray,
            device=device
        )
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), args.n_iters)
        summary_writer.add_scalar(
            'test/mse_all',
            np.mean(10.0 ** (-np.array(PSNRs_test) / 10.0)),
            args.n_iters
        )

    # final render‐path if requested
    if args.render_path:
        c2ws = test_dataset.render_path
        evaluation_path(
            test_dataset, tensorf, c2ws, renderer,
            f'{logfolder}/imgs_path_all/',
            N_vis=-1, N_samples=-1,
            white_bg=white_bg, ndc_ray=ndc_ray,
            device=device
        )


if __name__ == '__main__':
    args = config_parser()
    print(args)

    if args.export_mesh:
        export_mesh(args)
    elif args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)
