import pathlib
import numpy as np
import open3d as o3d
import torch

from datasets.nerf_st3d import SubjectLoader
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
        prior_occ = torch.exp(- dist_diff / (2*radius))
        return (
            x.cpu().numpy(),
            prior_occ.cpu().numpy(),
            i.cpu().numpy(),
            j.cpu().numpy(),
        )


data_root_fp = str(pathlib.Path.home() / "data/st3d")
scene = "03122_554516"

test_dataset = SubjectLoader(
    subject_id=scene,
    root_fp=data_root_fp,
    split="eval",
    num_rays=None,
    device="cpu",
    # **test_dataset_kwargs,
)
rgb = test_dataset.rgb
coord = test_dataset.coord

render_n_samples = 1024
render_step_size = 2 * np.sqrt(3) / render_n_samples
resolution = 256
print(2 / resolution)
print(render_step_size)
contraction_type = ContractionType.AABB
reshape_coord = coord.reshape(-1,3)
aabb = (reshape_coord.min(axis=0)-0.1).tolist() + (reshape_coord.max(axis=0)+0.1).tolist()

occupancy_grid = OccupancyGridWithPrior(
    roi_aabb=[-1, -1, -1, 1, 1, 1],
    resolution=resolution,
    contraction_type=contraction_type,
).to("cuda")
print("updating prior...")
x, prior_occ, i, j = occupancy_grid.update_prior_by_point_cloud(
    test_dataset.coord, render_step_size
)
mask = prior_occ>0.01
print(mask.shape)
print(mask.sum())
# assert False

valid_x = x[mask]
valid_rgb = rgb[i, j][mask]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(valid_x)
pcd.colors = o3d.utility.Vector3dVector(valid_rgb)
o3d.io.write_point_cloud("./data.ply", pcd)
