"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import os
import time
import pathlib
from typing import Callable

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from radiance_fields.mlp import VanillaNeRFRadianceField
from utils import render_image, set_random_seed

from nerfacc import ContractionType, OccupancyGrid, contract_inv


class OccupancyGridWithPrior(OccupancyGrid):
    def update_prior_by_point_cloud(
        self, point_cloud, radius, occ_thre: float = 0.01
    ):

        if isinstance(point_cloud, np.ndarray):
            point_cloud = torch.from_numpy(test_dataset.coord)
        point_cloud = point_cloud.to(self.grid_coords.device)

        # infer occupancy: density * step_size
        grid_coords = self.grid_coords
        x = grid_coords / self.resolution

        # voxel coordinates [0, 1]^3 -> world
        x = contract_inv(
            x,
            roi=self._roi_aabb,
            type=self._contraction_type,
        )

        # calculate x depth and norm
        x_dist = x.norm(dim=-1)
        x_norm = x / x_dist[:, None]

        # find raw indices
        _theta = torch.asin(x_norm[:, 1])
        _phi = torch.acos(x_norm[:, 0] / torch.cos(_theta))
        mask = x_norm[:, 2] > 0
        _phi[mask] *= -1
        H, W = point_cloud.shape[:2]
        i = torch.round((-(_theta * 2 / torch.pi) + 1) * H / 2).long() % H
        j = torch.round((0.5 - _phi / (2 * torch.pi)) * W).long() % W

        # calculate depth diff & occ
        p_dist = point_cloud.norm(dim=-1)
        dist_diff = torch.abs(p_dist[i, j] - x_dist)
        self.occs = torch.exp(- dist_diff / (2*radius))

        # self.register_buffer("prior_occ", prior_occ)
        self._binary = (self.occs > occ_thre).view(self._binary.shape)

    @torch.no_grad()
    def _update(
        self,
        step: int,
        occ_eval_fn: Callable,
        occ_thre: float = 0.01,
        ema_decay: float = 0.95,
        warmup_steps: int = 256,
    ) -> None:
        """Update the occ field in the EMA way."""
        # sample cells
        if step < warmup_steps: #freeze the self._binary for the warmup stage
            return
        #     indices = self._get_all_cells()
        # else:
        N = self.num_cells // (4 * 8) # add 8 for 256 
        indices = self._sample_uniform_and_occupied_cells(N)

        # infer occupancy: density * step_size
        grid_coords = self.grid_coords[indices]
        x = (
            grid_coords + torch.rand_like(grid_coords, dtype=torch.float32)
        ) / self.resolution
        if self._contraction_type == ContractionType.UN_BOUNDED_SPHERE:
            # only the points inside the sphere are valid
            mask = (x - 0.5).norm(dim=1) < 0.5
            x = x[mask]
            indices = indices[mask]
        # voxel coordinates [0, 1]^3 -> world
        x = contract_inv(
            x,
            roi=self._roi_aabb,
            type=self._contraction_type,
        )
        occ = occ_eval_fn(x).squeeze(-1)

        # ema update
        # v0 change
        # self.occs[indices] = torch.maximum(
        #     self.prior_occ[indices],
        #     torch.maximum(self.occs[indices] * ema_decay, occ),
        # )
        # if step < warmup_steps:
        #     ratio = step / warmup_steps
        #     self.occs[indices] = (
        #         self.prior_occ[indices] * (1 - ratio)
        #         + self.occs[indices] * ratio
        #     )
        self.occs[indices] = torch.maximum(self.occs[indices] * ema_decay, occ)
        
        # suppose to use scatter max but emperically it is almost the same.
        # self.occs, _ = scatter_max(
        #     occ, indices, dim=0, out=self.occs * ema_decay
        # )
        # TODO(yang): remove mean self.occs.mean()
        self._binary = (
            self.occs > torch.clamp(self.occs.mean(), max=occ_thre)
        ).view(self._binary.shape)


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
        choices=[
            "03122_554516",
        ],
        help="which scene to use",
    )
    # parser.add_argument(
    #     "--aabb",
    #     type=lambda s: [float(item) for item in s.split(",")],
    #     default="-1.,-1.,-1.,1.,1.,1.",
    #     help="delimited list input",
    # )
    parser.add_argument(
        "--test_chunk_size",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--unbounded",
        action="store_true",
        help="whether to use unbounded rendering",
    )
    parser.add_argument("--cone_angle", type=float, default=0.0)

    # add depth loss
    parser.add_argument(
        "--add_depth_loss", action="store_true", help="use depth for loss"
    )
    parser.add_argument(
        "--add_depth_prior",
        action="store_true",
        help="use depth for occupy update",
    )
    args = parser.parse_args()

    logdir = pathlib.Path("./logs/") / args.expname / args.scene
    logdir.mkdir(parents=True, exist_ok=True)

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

    # initialize logs
    with open(logdir / "statistics.txt", "w") as f:
        pass

    # load checkpoint
    # ckpt_path = "logs/model.ckpt"
    # ckpt = torch.load(ckpt_path)
    # radiance_field.load_state_dict(ckpt)

    # setup the dataset
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}

    from datasets.nerf_st3d import SubjectLoader

    data_root_fp = str(pathlib.Path.home() / "data/st3d")
    target_sample_batch_size = 1 << 17
    grid_resolution = 256
    render_n_samples = 1024

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
        coord = test_dataset.coord.reshape(-1,3)
        aabb = (coord.min(axis=0)-0.1).tolist() + (coord.max(axis=0)+0.1).tolist()
        scene_aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        near_plane = None
        far_plane = None
        render_step_size = (
            (scene_aabb[3:] - scene_aabb[:3]).max()
            * math.sqrt(3)
            / render_n_samples
        ).item()

    # adding depth prior for occupancy_grid
    if args.add_depth_prior:
        occupancy_grid = OccupancyGridWithPrior(
            roi_aabb=aabb,
            resolution=grid_resolution,
            contraction_type=contraction_type,
        ).to(device)
        print("updating prior...")
        occupancy_grid.update_prior_by_point_cloud(
            test_dataset.coord, render_step_size
        )
    else:
        occupancy_grid = OccupancyGrid(
            roi_aabb=aabb,
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
            rgb_loss = F.smooth_l1_loss(
                rgb[alive_ray_mask], pixels[alive_ray_mask]
            )
            loss = rgb_loss

            if args.add_depth_loss:
                depth_loss = F.smooth_l1_loss(
                    depth[alive_ray_mask], data["depth"][alive_ray_mask]
                )
                loss += depth_loss

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
                if args.add_depth_loss:
                    reprs += f" rgb_loss={rgb_loss:.5f} | depth_loss={depth_loss:.5f} |"
                print(reprs)
                with open(logdir / "statistics.txt", "a") as f:
                    f.write(reprs + "\n")

            if step >= 0 and step % eval_steps == 0 and step > 0:
                test_dir = logdir / f"test_{step:08d}"
                test_dir.mkdir(parents=True, exist_ok=True)

                # evaluation
                radiance_field.eval()

                psnrs = []
                depth_errors = []
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
                        depth_error = F.smooth_l1_loss(depth, data["depth"])
                        depth_errors.append(depth_error.item())
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
                depth_error_avg = sum(depth_errors) / len(depth_errors)
                reprs = f"evaluation: psnr_avg={psnr_avg} | depth_error_avg={depth_error_avg}"
                print(reprs)
                with open(logdir / "statistics.txt", "a") as f:
                    f.write(reprs + "\n")
                torch.save(
                    radiance_field.state_dict(),
                    test_dir / f"model-psnr-{psnr_avg}.ckpt",
                )

            if step == max_steps:
                print("training stops")
                exit()

            step += 1
