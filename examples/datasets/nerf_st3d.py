import collections
import json
import os

from PIL import Image
import os
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .utils import Rays


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "test", "eval"]
    SUBJECT_IDS = [
        "03122_554516",
    ]

    # WIDTH, HEIGHT = 1024, 512
    NEAR, FAR = 0.0, 2.0

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        device: str,
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        self.baseDir = os.path.join(root_fp, subject_id)
        self.split = split
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        # self.training = (num_rays is not None) and (split in ["train"])
        self.batch_over_images = batch_over_images
        self.device = device
        
        self._load_raw_data()

    def __len__(self):
        return self.rays_indices.shape[0] - 1

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    @torch.no_grad()
    def __getitem__(self, index):
        num_rays = self.num_rays

        if self.split == "eval":
            indices = np.arange(
                self.rays_indices[index], self.rays_indices[index + 1]
            )
            origins = torch.reshape(
                torch.from_numpy(self.rays_o[indices]), (self.H, self.W, 3)
            ).to(self.device)
            viewdirs = torch.reshape(
                torch.from_numpy(self.rays_d[indices]), (self.H, self.W, 3)
            ).to(self.device)
            rgb = torch.reshape(
                torch.from_numpy(self.rays_rgb[indices]),
                (self.H, self.W, 3),
            ).to(self.device)
        else:
            if self.batch_over_images:
                indices = np.random.randint(
                    0,
                    self.rays_o.shape[0],
                    size=(num_rays,),
                )
            else:
                indices = np.random.randint(
                    self.rays_indices[index],
                    self.rays_indices[index + 1],
                    size=(num_rays,),
                )

            origins = torch.reshape(
                torch.from_numpy(self.rays_o[indices]), (num_rays, 3)
            ).to(self.device)
            viewdirs = torch.reshape(
                torch.from_numpy(self.rays_d[indices]), (num_rays, 3)
            ).to(self.device)
            rgb = torch.reshape(
                torch.from_numpy(self.rays_rgb[indices]), (num_rays, 3)
            ).to(self.device)

        rays = Rays(origins=origins, viewdirs=viewdirs)
        return {
            "pixels": rgb,  # [h, w, 3] or [num_rays, 3]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "color_bkgd": torch.zeros(3, device=self.device)
        }

    def _load_raw_data(self):
        """load the raw data from files"""
        # load rgb image
        baseDir = self.baseDir
        basename = baseDir.split("/")[-1] + "_"
        rgb = (
            np.asarray(
                Image.open(os.path.join(baseDir, basename + "rgb.png")).convert(
                    "RGB"
                )
            )
            / 255.0
        )

        # load depth image
        if baseDir.split("/")[-2] == "mp3d":
            print(os.path.join(baseDir, basename + "depth.exr"))
            d = cv2.imread(
                os.path.join(baseDir, basename + "depth.exr"),
                cv2.IMREAD_ANYDEPTH,
            )
            d = d.astype(np.float)

        else:
            d = np.asarray(
                Image.open(os.path.join(baseDir, basename + "d.png"))
            )

        # resize rgb image to depth size
        H, W = d.shape[0], d.shape[1]
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

        # calculate the gradient based on the rgb images
        gradient = cv2.Laplacian(rgb, cv2.CV_64F)
        gradient = 2 * (gradient - np.min(gradient)) / np.ptp(gradient) - 1

        # normalize to [0, 1]
        max_depth = np.max(d)
        d = d.reshape(rgb.shape[0], rgb.shape[1], 1) / max_depth

        _y = np.repeat(np.array(range(W)).reshape(1, W), H, axis=0)
        _x = np.repeat(np.array(range(H)).reshape(1, H), W, axis=0).T

        # change the pixel coordinate to the polar coordinate
        _theta = (1 - 2 * (_x) / H) * np.pi / 2  # latitude
        _phi = 2 * math.pi * (0.5 - (_y) / W)  # longtitude

        axis0 = (np.cos(_theta) * np.cos(_phi)).reshape(H, W, 1)
        axis1 = np.sin(_theta).reshape(H, W, 1)
        axis2 = (-np.cos(_theta) * np.sin(_phi)).reshape(H, W, 1)
        original_coord = np.concatenate((axis0, axis1, axis2), axis=2)
        coord = original_coord * d  # add depth to each pixel

        # load training camera poses
        if self.split == "eval":
            origins = np.array([0.0, 0.0, 0.0]).reshape(1, 3)
            self.rays_o = np.repeat(
                origins.reshape(1, -1), H * W, axis=0
            ).reshape(-1, 3).astype(np.float32)
            self.rays_d = original_coord.reshape(-1, 3).astype(np.float32)
            self.rays_g = gradient.reshape(-1, 3).astype(np.float32)
            self.rays_rgb = rgb.reshape(-1, 3).astype(np.float32)
            self.rays_depth = d.reshape(-1, 1).astype(np.float32)
            self.rays_indices = np.array([0, H * W], dtype=int)

        else:
            origins = []
            with open(
                os.path.join(baseDir, self.split, "cam_pos.txt"), "r"
            ) as fp:
                all_poses = fp.readlines()
                for p in all_poses:
                    origins.append(np.array(p.split()).astype(float))

            indices, rays_indices = 0, [0]
            rays_o, rays_d, rays_g, rays_rgb, rays_depth = [], [], [], [], []
            for i, o in enumerate(origins):
                # coord is a spherical projection, now from p depth is calculated
                dep = np.linalg.norm(coord - o, axis=-1)

                # direction = end point - start point
                viewdir = coord - o
                viewdir = viewdir / np.linalg.norm(viewdir, axis=-1)[..., None]

                # get mask
                mask = (
                    np.asarray(
                        Image.open(
                            os.path.join(baseDir, self.split, "mask_%d.png" % i)
                        )
                    )
                    / 255
                )

                indices = (mask > 0).sum()
                rays_o.append(np.repeat(o.reshape(1, -1), indices, axis=0))
                rays_g.append(gradient[mask > 0])
                rays_d.append(viewdir[mask > 0])
                rays_rgb.append(rgb[mask > 0])
                rays_depth.append(dep[mask > 0])
                rays_indices.append(indices)

            # get output
            self.rays_o = np.concatenate(rays_o, axis=0).astype(np.float32)
            self.rays_d = np.concatenate(rays_d, axis=0).astype(np.float32)
            self.rays_g = np.concatenate(rays_g, axis=0).astype(np.float32)
            self.rays_rgb = np.concatenate(rays_rgb, axis=0).astype(np.float32)
            self.rays_depth = np.concatenate(rays_depth, axis=0).astype(np.float32)
            self.rays_indices = np.cumsum(rays_indices, dtype=int).astype(np.float32)
        self.coord = coord
        self.H, self.W = H, W
