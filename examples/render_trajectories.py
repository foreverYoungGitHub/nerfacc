import functools
import os
from PIL import Image
import numpy as np
import torch

from radiance_fields.mlp import VanillaNeRFRadianceField
from nerfacc import OccGridEstimator
from utils import Rays, render_image_with_occgrid
from pose import Pose, to_homogeneous, Camera
from tqdm import tqdm

def load_nerf_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    # load the nerf model
    state_dict = torch.load(model_path)

    radiance_field = VanillaNeRFRadianceField().to(device)
    radiance_field.load_state_dict(state_dict["radiance_field_state_dict"])

    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    occupancy_grid = OccGridEstimator(aabb).to(device)
    occupancy_grid.load_state_dict(state_dict["estimator_state_dict"])
    radiance_field.eval()
    occupancy_grid.eval()

    render_func = functools.partial(
        render_image_with_occgrid,
        radiance_field=radiance_field,
        estimator=occupancy_grid,
        render_step_size=5e-3,
        render_bkgd=torch.zeros(3, device=device),
        test_chunk_size=8192,
    )
    return render_func

def render_image(T_cam2w, camera, render_func) -> Image:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    center = T_cam2w.t
    p2d = np.mgrid[:camera.height, :camera.width].reshape(2, -1)[::-1].T.astype(np.float32)
    origins = np.tile(center.astype(np.float32)[None], (len(p2d), 1))
    p2d += 0.5  # to COLMAP coordinates

    p2d_norm = camera.image2world(p2d)
    R = T_cam2w.R
    # It is much faster to perform the transformation in fp32
    directions = to_homogeneous(p2d_norm.astype(np.float32)) @ R.astype(np.float32).T

    # Outputs must be contiguous.
    origins = np.ascontiguousarray(origins, dtype=np.float32)
    directions = np.ascontiguousarray(directions, dtype=np.float32)

    origins = origins.reshape(camera.height, camera.width, 3)
    origins = torch.from_numpy(origins).to(device)
    directions = directions.reshape(camera.height, camera.width, 3)
    directions = torch.from_numpy(directions).to(device)

    rays = Rays(origins=origins, viewdirs=directions)

    with torch.no_grad():
        rgb, acc, depth, _ =render_func(rays=rays)
        # flip image
        rgb = torch.flip(rgb, dims=(0,1))
    return Image.fromarray(
                            (rgb.cpu().numpy() * 255).astype(np.uint8)
                        )

# Main script
if __name__ == "__main__":
    render_func = load_nerf_model("./logs/st3d_gd/492165/test_00020000/model-psnr-28.36334228515625.ckpt")
    render_dir = "./logs/st3d_gd/492165/test_00020000/rendered_images"
    if not os.path.exists(render_dir):
        os.makedirs(render_dir)

    camera = Camera(
        'SIMPLE_PINHOLE', [480, 480, 100, 240, 240]
    )

    # Set image resolution


    # Define camera parameters (replace with your camera info)
    T_cam2w = Pose(
        r=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        t=np.array([0, 0, 0])
    )

    # create trajectory
    # first move forward 1m with 30 steps
    # then rotate 90 degree with 30 steps
    # then move back 1m
    # then rotate 90 degree
    # then move forward 1m
    # then rotate 90 degree
    # then move back 1m
    # then rotate 90 degree

    traj = []
    step = 10
    for i in range(step):
        T_cam2w = T_cam2w * Pose(t=np.array([0, 0, 0.05/step]))
        traj.append(T_cam2w.clone())
    for i in range(step):
        # rotate 3 degree in y axis
        degree = (90/step)/180*np.pi
        T_cam2w = T_cam2w * Pose(r=np.array([[np.cos(degree), 0, np.sin(degree)], [0, 1, 0], [-np.sin(degree), 0, np.cos(degree)]]))
        traj.append(T_cam2w.clone())
    for i in range(step):
        T_cam2w = T_cam2w * Pose(t=np.array([0, 0, -0.05/step]))
        traj.append(T_cam2w.clone())
    for i in range(step):
        # rotate 3 degree in y axis
        degree = (90/step)/180*np.pi
        T_cam2w = T_cam2w * Pose(r=np.array([[np.cos(degree), 0, np.sin(degree)], [0, 1, 0], [-np.sin(degree), 0, np.cos(degree)]]))
        traj.append(T_cam2w.clone())
    for i in range(step):
        T_cam2w = T_cam2w * Pose(t=np.array([0, 0, 0.05/step]))
        traj.append(T_cam2w.clone())
    for i in range(step):
        # rotate 3 degree in y axis
        degree = (90/step)/180*np.pi
        T_cam2w = T_cam2w * Pose(r=np.array([[np.cos(degree), 0, np.sin(degree)], [0, 1, 0], [-np.sin(degree), 0, np.cos(degree)]]))
        traj.append(T_cam2w.clone())
    for i in range(step):
        T_cam2w = T_cam2w * Pose(t=np.array([0, 0, -0.05/step]))
        traj.append(T_cam2w.clone())
    for i in range(step):
        # rotate 3 degree in y axis
        degree = (90/step)/180*np.pi
        T_cam2w = T_cam2w * Pose(r=np.array([[np.cos(degree), 0, np.sin(degree)], [0, 1, 0], [-np.sin(degree), 0, np.cos(degree)]]))
        traj.append(T_cam2w.clone())
    

    # for i in tqdm(range(len(traj))):
    #     T_cam2w = traj[i]
    #     rendered_image = render_image(T_cam2w, camera, render_func)
    #     rendered_image.save(render_dir+f"/{i:04d}.png")

    # move image to 04d.png
    for i in tqdm(range(len(traj))):
        image_path = render_dir+f"/{i}.png"
        new_image_path = render_dir+f"/{i:04d}.png"
        os.rename(image_path, new_image_path)
