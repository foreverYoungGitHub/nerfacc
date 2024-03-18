from datasets.nerf_st3d import SubjectLoader
import torch
import numpy as np

from radiance_fields.mlp import VanillaNeRFRadianceField
from nerfacc import OccGridEstimator
from utils import render_image_with_occgrid

# def generate_point_cloud(
#     pipeline: Pipeline,
#     num_points: int = 1000000,
#     remove_outliers: bool = True,
#     estimate_normals: bool = False,
#     reorient_normals: bool = False,
#     rgb_output_name: str = "rgb",
#     depth_output_name: str = "depth",
#     normal_output_name: Optional[str] = None,
#     use_bounding_box: bool = True,
#     bounding_box_min: Optional[Tuple[float, float, float]] = None,
#     bounding_box_max: Optional[Tuple[float, float, float]] = None,
#     crop_obb: Optional[OrientedBox] = None,
#     std_ratio: float = 10.0,
# ) -> o3d.geometry.PointCloud:
#     """Generate a point cloud from a nerf.

#     Args:
#         pipeline: Pipeline to evaluate with.
#         num_points: Number of points to generate. May result in less if outlier removal is used.
#         remove_outliers: Whether to remove outliers.
#         reorient_normals: Whether to re-orient the normals based on the view direction.
#         estimate_normals: Whether to estimate normals.
#         rgb_output_name: Name of the RGB output.
#         depth_output_name: Name of the depth output.
#         normal_output_name: Name of the normal output.
#         use_bounding_box: Whether to use a bounding box to sample points.
#         bounding_box_min: Minimum of the bounding box.
#         bounding_box_max: Maximum of the bounding box.
#         std_ratio: Threshold based on STD of the average distances across the point cloud to remove outliers.

#     Returns:
#         Point cloud.
#     """

#     progress = Progress(
#         TextColumn(":cloud: Computing Point Cloud :cloud:"),
#         BarColumn(),
#         TaskProgressColumn(show_speed=True),
#         TimeRemainingColumn(elapsed_when_finished=True, compact=True),
#         console=CONSOLE,
#     )
#     points = []
#     rgbs = []
#     normals = []
#     view_directions = []
#     if use_bounding_box and (crop_obb is not None and bounding_box_max is not None):
#         CONSOLE.print("Provided aabb and crop_obb at the same time, using only the obb", style="bold yellow")
#     with progress as progress_bar:
#         task = progress_bar.add_task("Generating Point Cloud", total=num_points)
#         while not progress_bar.finished:
#             normal = None

#             with torch.no_grad():
#                 ray_bundle, _ = pipeline.datamanager.next_train(0)
#                 assert isinstance(ray_bundle, RayBundle)
#                 outputs = pipeline.model(ray_bundle)
#             if rgb_output_name not in outputs:
#                 CONSOLE.rule("Error", style="red")
#                 CONSOLE.print(f"Could not find {rgb_output_name} in the model outputs", justify="center")
#                 CONSOLE.print(f"Please set --rgb_output_name to one of: {outputs.keys()}", justify="center")
#                 sys.exit(1)
#             if depth_output_name not in outputs:
#                 CONSOLE.rule("Error", style="red")
#                 CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
#                 CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
#                 sys.exit(1)
#             rgba = pipeline.model.get_rgba_image(outputs, rgb_output_name)
#             depth = outputs[depth_output_name]
#             if normal_output_name is not None:
#                 if normal_output_name not in outputs:
#                     CONSOLE.rule("Error", style="red")
#                     CONSOLE.print(f"Could not find {normal_output_name} in the model outputs", justify="center")
#                     CONSOLE.print(f"Please set --normal_output_name to one of: {outputs.keys()}", justify="center")
#                     sys.exit(1)
#                 normal = outputs[normal_output_name]
#                 assert (
#                     torch.min(normal) >= 0.0 and torch.max(normal) <= 1.0
#                 ), "Normal values from method output must be in [0, 1]"
#                 normal = (normal * 2.0) - 1.0
#             point = ray_bundle.origins + ray_bundle.directions * depth
#             view_direction = ray_bundle.directions

#             # Filter points with opacity lower than 0.5
#             mask = rgba[..., -1] > 0.5
#             point = point[mask]
#             view_direction = view_direction[mask]
#             rgb = rgba[mask][..., :3]
#             if normal is not None:
#                 normal = normal[mask]

#             if use_bounding_box:
#                 if crop_obb is None:
#                     comp_l = torch.tensor(bounding_box_min, device=point.device)
#                     comp_m = torch.tensor(bounding_box_max, device=point.device)
#                     assert torch.all(
#                         comp_l < comp_m
#                     ), f"Bounding box min {bounding_box_min} must be smaller than max {bounding_box_max}"
#                     mask = torch.all(torch.concat([point > comp_l, point < comp_m], dim=-1), dim=-1)
#                 else:
#                     mask = crop_obb.within(point)
#                 point = point[mask]
#                 rgb = rgb[mask]
#                 view_direction = view_direction[mask]
#                 if normal is not None:
#                     normal = normal[mask]

#             points.append(point)
#             rgbs.append(rgb)
#             view_directions.append(view_direction)
#             if normal is not None:
#                 normals.append(normal)
#             progress.advance(task, point.shape[0])
#     points = torch.cat(points, dim=0)
#     rgbs = torch.cat(rgbs, dim=0)
#     view_directions = torch.cat(view_directions, dim=0).cpu()

#     import open3d as o3d

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
#     pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())

#     ind = None
#     if remove_outliers:
#         CONSOLE.print("Cleaning Point Cloud")
#         pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
#         print("\033[A\033[A")
#         CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")
#         if ind is not None:
#             view_directions = view_directions[ind]

#     # either estimate_normals or normal_output_name, not both
#     if estimate_normals:
#         if normal_output_name is not None:
#             CONSOLE.rule("Error", style="red")
#             CONSOLE.print("Cannot estimate normals and use normal_output_name at the same time", justify="center")
#             sys.exit(1)
#         CONSOLE.print("Estimating Point Cloud Normals")
#         pcd.estimate_normals()
#         print("\033[A\033[A")
#         CONSOLE.print("[bold green]:white_check_mark: Estimating Point Cloud Normals")
#     elif normal_output_name is not None:
#         normals = torch.cat(normals, dim=0)
#         if ind is not None:
#             # mask out normals for points that were removed with remove_outliers
#             normals = normals[ind]
#         pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())

#     # re-orient the normals
#     if reorient_normals:
#         normals = torch.from_numpy(np.array(pcd.normals)).float()
#         mask = torch.sum(view_directions * normals, dim=-1) > 0
#         normals[mask] *= -1
#         pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())

#     return pcd


# convert the nerf model to a 3D model
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the nerf model
state_dict = torch.load("./logs/st3d_gd/492165/test_00020000/model-psnr-28.36334228515625.ckpt")


radiance_field = VanillaNeRFRadianceField().to(device)
radiance_field.load_state_dict(state_dict["radiance_field_state_dict"])

aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
occupancy_grid = OccGridEstimator(aabb).to(device)
occupancy_grid.load_state_dict(state_dict["estimator_state_dict"])
radiance_field.eval()
occupancy_grid.eval()

del state_dict
torch.cuda.empty_cache()

data_root_fp = "/home/jupyter/data"

test_dataset = SubjectLoader(
    subject_id="492165",
    root_fp=data_root_fp,
    split="eval",
    num_rays=None,
    device=device,
    batch_over_images=False,
    with_mask=False,
)

render_step_size = 5e-3

points = []
rgbs = []
view_directions = []
with torch.no_grad():
    for i in range(len(test_dataset)):
        print(f"Processing {i+1}/{len(test_dataset)}")
        data = test_dataset[i]
        rays = data["rays"]
        render_bkgd = data["color_bkgd"]

        # rendering
        rgb, acc, depth, _ = render_image_with_occgrid(
            radiance_field,
            occupancy_grid,
            rays,
            # scene_aabb,
            # rendering options
            # near_plane=None,
            # far_plane=None,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            # cone_angle=args.cone_angle,
            # test options
            test_chunk_size=4096,
        )

        point = rays.origins + rays.viewdirs * depth
        view_direction = rays.viewdirs

        # Filter points with opacity lower than 0.5
        mask = acc > 0.5
        mask = mask[:,:,0]
        point = point[mask]
        view_direction = view_direction[mask]
        rgb = rgb[mask]

        points.append(point)
        rgbs.append(rgb)
        view_directions.append(view_direction)

points = torch.cat(points, dim=0)
rgbs = torch.cat(rgbs, dim=0)
view_directions = torch.cat(view_directions, dim=0).cpu()

import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())
pcd.estimate_normals()
#dump the point cloud to a file
o3d.io.write_point_cloud("nerf_point_cloud.ply", pcd)

print("Computing Mesh... this may take a while.")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
vertices_to_remove = densities < np.quantile(densities, 0.1)
mesh.remove_vertices_by_mask(vertices_to_remove)

# dump the mesh to a file
o3d.io.write_triangle_mesh("nerf_mesh.ply", mesh)