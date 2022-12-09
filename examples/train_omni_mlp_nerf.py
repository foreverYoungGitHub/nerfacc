"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import os
import time
import pathlib

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from radiance_fields.mlp import VanillaNeRFRadianceField
from utils import render_image, set_random_seed

from nerfacc import ContractionType, OccupancyGrid

if __name__ == "__main__":

    device = "cuda:0"
    set_random_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        choices=["train", "trainval"],
        help="which train split to use",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="03122_554516",
        choices=["03122_554516",],
        help="which scene to use",
    )
    parser.add_argument(
        "--aabb",
        type=lambda s: [float(item) for item in s.split(",")],
        default="-1.,-1.,-1.,1.,1.,1.",
        help="delimited list input",
    )
    parser.add_argument(
        "--test_chunk_size", type=int, default=8192,
    )
    parser.add_argument(
        "--unbounded",
        action="store_true",
        help="whether to use unbounded rendering",
    )
    parser.add_argument("--cone_angle", type=float, default=0.0)
    args = parser.parse_args()

    render_n_samples = 1024

    logdir = pathlib.Path("./logs/") / args.expname / args.scene
    logdir.mkdir(parents=True, exist_ok=True)

    # setup the scene bounding box.
    if args.unbounded:
        print("Using unbounded rendering")
        contraction_type = ContractionType.UN_BOUNDED_SPHERE
        # contraction_type = ContractionType.UN_BOUNDED_TANH
        scene_aabb = None
        near_plane = 0.2
        far_plane = 1e4
        render_step_size = 1e-2
    else:
        contraction_type = ContractionType.AABB
        scene_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
        near_plane = None
        far_plane = None
        render_step_size = (
            (scene_aabb[3:] - scene_aabb[:3]).max()
            * math.sqrt(3)
            / render_n_samples
        ).item()

    # setup the radiance field we want to train.
    max_steps = 100000
    eval_steps = 10000
    grad_scaler = torch.cuda.amp.GradScaler(1)
    radiance_field = VanillaNeRFRadianceField().to(device)
    optimizer = torch.optim.Adam(radiance_field.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            max_steps // 2,
            max_steps * 3 // 4,
            max_steps * 5 // 6,
            max_steps * 9 // 10,
        ],
        gamma=0.33,
    )
    
    # load checkpoint
    ckpt_path = "logs/model.ckpt"
    ckpt = torch.load(ckpt_path)
    # print(ckpt)
    radiance_field.load_state_dict(ckpt)

    # setup the dataset
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}

    from datasets.nerf_st3d import SubjectLoader

    data_root_fp = str(pathlib.Path.home() / "data/st3d")
    target_sample_batch_size = 1 << 17
    grid_resolution = 128

    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split=args.train_split,
        num_rays=target_sample_batch_size // render_n_samples,
        device=device,
        **train_dataset_kwargs,
    )

    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split="eval",
        num_rays=None,
        device=device,
        **test_dataset_kwargs,
    )

    occupancy_grid = OccupancyGrid(
        roi_aabb=args.aabb,
        resolution=grid_resolution,
        contraction_type=contraction_type,
    ).to(device)

    # training
    step = 0
    tic = time.time()
    for epoch in range(10000000):
        for i in range(len(train_dataset)):
            radiance_field.train()
            data = train_dataset[i]

            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

            # update occupancy grid
            occupancy_grid.every_n_step(
                step=step,
                occ_eval_fn=lambda x: radiance_field.query_opacity(
                    x, render_step_size
                ),
            )

            # render
            rgb, acc, depth, n_rendering_samples = render_image(
                radiance_field,
                occupancy_grid,
                rays,
                scene_aabb,
                # rendering options
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=args.cone_angle,
            )
            if n_rendering_samples == 0:
                continue

            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int(
                num_rays
                * (target_sample_batch_size / float(n_rendering_samples))
            )
            train_dataset.update_num_rays(num_rays)
            alive_ray_mask = acc.squeeze(-1) > 0

            # compute loss
            loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])

            optimizer.zero_grad()
            # do not unscale it because we are using Adam.
            grad_scaler.scale(loss).backward()
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                elapsed_time = time.time() - tic
                loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                reprs = (
                    f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                    f"loss={loss:.5f} | "
                    f"alive_ray_mask={alive_ray_mask.long().sum():d} | "
                    f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} |"
                )
                print(reprs)
                with open(logdir/"statistics.txt", "a") as f:
                    f.write(reprs + "\n")

            if step >= 0 and step % eval_steps == 0 and step > 0:
                test_dir = logdir / f"test_{step:08d}"
                test_dir.mkdir(parents=True, exist_ok=True)
                
                # evaluation
                radiance_field.eval()

                psnrs = []
                with torch.no_grad():
                    for i in tqdm.tqdm(range(len(test_dataset))):
                        data = test_dataset[i]
                        render_bkgd = data["color_bkgd"]
                        rays = data["rays"]
                        pixels = data["pixels"]

                        # rendering
                        rgb, acc, depth, _ = render_image(
                            radiance_field,
                            occupancy_grid,
                            rays,
                            scene_aabb,
                            # rendering options
                            near_plane=None,
                            far_plane=None,
                            render_step_size=render_step_size,
                            render_bkgd=render_bkgd,
                            cone_angle=args.cone_angle,
                            # test options
                            test_chunk_size=args.test_chunk_size,
                        )
                        mse = F.mse_loss(rgb, pixels)
                        psnr = -10.0 * torch.log(mse) / np.log(10.0)
                        psnrs.append(psnr.item())
                        imageio.imwrite(
                            test_dir / "acc_binary_test.png",
                            ((acc > 0).float().cpu().numpy() * 255).astype(
                                np.uint8
                            ),
                        )
                        imageio.imwrite(
                            test_dir / "rgb_test.png",
                            (rgb.cpu().numpy() * 255).astype(np.uint8),
                        )
                        # break
                psnr_avg = sum(psnrs) / len(psnrs)
                print(f"evaluation: psnr_avg={psnr_avg}")
                with open(logdir/"statistics.txt", "a") as f:
                    f.write(f"evaluation: psnr_avg={psnr_avg} \n")
                torch.save(
                    radiance_field.state_dict(), test_dir / f"model-psnr-{psnr_avg}.ckpt",
                )

            if step == max_steps:
                print("training stops")
                exit()

            step += 1
