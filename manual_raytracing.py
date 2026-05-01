# TODO: Load Gaussian Model, be able to render it with a basic camera position
# From there, be able to intersect rays with a sphere, 

import os
from pathlib import Path
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
import sys
from scene import Scene, GaussianModel, camera_to_JSON
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from icecream import ic
import random
import math
import cv2
import json
import traceback
from utils.system_utils import searchForMaxIteration
import time
from gaussian_renderer.fast_renderer import FastRenderer
from gaussian_renderer import render, network_gui
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

from scene.dataset_readers import ProjectionType

# Graphing
import matplotlib
matplotlib.use('Agg') # headless mode
import matplotlib.pyplot as plt
import numpy as np

def load_gaussian_model(model_params: ModelParams, opt_params: OptimizationParams, checkpoint: os.PathLike | None) -> GaussianModel:
    first_iter = 0
    gaussians = GaussianModel(model_params.sh_degree, model_params.use_neural_network, model_params.max_opacity)

    if checkpoint is not None:
        (gaussian_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(gaussian_params, opt_params)

    # Search for a valid EVER-comaptible .ply
    else:
        loaded_iter = searchForMaxIteration(os.path.join(model_params.model_path, "point_cloud"))
        print("Loading trained model at iteration {}".format(loaded_iter))

        gaussians.load_ply(os.path.join(model_params.model_path,
                                                       "point_cloud",
                                                       "iteration_" + str(loaded_iter),
                                                       "point_cloud.ply"))
        
    return gaussians

def render_gaussian_model(gaussians: GaussianModel, model_params: ModelParams, opt_params: OptimizationParams, pipe_params: PipelineParams) -> torch.Tensor:
    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians.training_setup(opt_params)
    torch.cuda.empty_cache()

    camera_index = 31
    # Get the cameras.json file that enumerates all the cameras and their positions
    rendering_cam = get_rendering_cam(model_params, camera_index)
    
    # TODO: Do rendering
    preview_factor = 250
    image_width = rendering_cam.image_width
    image_height = rendering_cam.image_height

    rendering_cam.image_width = image_width // preview_factor
    rendering_cam.image_height = image_height // preview_factor

    print(f"{rendering_cam = }")

    renderer = FastRenderer(rendering_cam, gaussians, pipe_params.enable_GLO)
    renderer.set_camera(rendering_cam)

    #Render sphere
    rays_o, rays_d = renderer.get_rays(rendering_cam)

    sphere_center_translation = torch.zeros((1, 3), device="cuda")
    avg_position = torch.mean(rays_o, dim=0).reshape(1, 3)
    avg_direction = torch.mean(rays_d, dim=0).reshape(1, 3)

    sphere_center = avg_position + avg_direction * 2
    sphere_radius = .5
    
    T_vals = intersect_sphere(rays_o, rays_d, sphere_center, sphere_radius) # (h*w) x 1
    bounce_ray_o, bounce_ray_d = bounce_off_sphere(rays_o, rays_d, T_vals, sphere_center)

    plot_rays_and_sphere(rays_o, rays_d, sphere_center, sphere_radius, T_vals, bounce_ray_o, bounce_ray_d)

    # TODO: Can use rendering t_max for creating sphere?
    bounced_ray_output = renderer.trace_rays(bounce_ray_o, bounce_ray_d, rendering_cam, 0, 1e7)
    bounce_image = bounced_ray_output['color'][:, :3].T.reshape(3, rendering_cam.image_height, rendering_cam.image_width)

    T_vals = T_vals.reshape(1, rendering_cam.image_height, rendering_cam.image_width) # (1, h, w)

    # Render quickly
    st = time.time()

    # TODO: Use opposite of t_min (t_max?) for creating chromesphere?
    # t_max parameter would need to become a tensor basically...kinda weird
    image = renderer.render(rendering_cam, gaussians, background) # (3, h, w)

    # Add bounce lighting to the image
    masked_image = image * torch.isinf(T_vals) # Mask out part that hits the sphere
    image = masked_image + bounce_image # Add in bounce lighting
    
    print(f"{image.shape}")

    print(f"Took {time.time()-st}s to render frame.")

    # Convert to 8-bit and save as png
    image = (torch.clamp(image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
    image = cv2.resize(image, (image_width, image_height))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.imwrite(Path(model_params.model_path) / "test_output.png", image)
    
    torch.cuda.empty_cache()
    return image

def plot_rays_and_sphere(rays_o: torch.Tensor, rays_d: torch.Tensor, sphere_center: torch.Tensor, sphere_radius: torch.Tensor, 
                         T_vals: torch.Tensor, bounce_ray_o: torch.Tensor, bounce_ray_d: torch.Tensor):
    """
    Create matplotlib plots of the original camera rays, sphere, and the rays that bounced off of it.
    """
    ax = plot_ray_o_and_d(torch.cat([rays_o[0:1, :], bounce_ray_o]), torch.cat([torch.zeros_like(rays_d[0:1, :]), bounce_ray_d]), 
                          render_fig=False, alpha=.5, label="Bounced rays")
    sphere_center_list = sphere_center.cpu().ravel().tolist()
    
    
    # plt.axis('equal')
    ax.set_aspect('equal')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    # Make sphere data (from https://matplotlib.org/stable/gallery/mplot3d/surface3d_2.html)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = sphere_radius * np.outer(np.cos(u), np.sin(v)) + sphere_center_list[0]
    y = sphere_radius * np.outer(np.sin(u), np.sin(v)) + sphere_center_list[1]
    z = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + sphere_center_list[2]

    ax.plot_surface(x, y, z, alpha=.7, color='y', label='Sphere')
    ax.scatter(*sphere_center_list, marker="X", s=60, label='Sphere Center')

    # Show bouncing rays:
    bouncing_ray_indices = torch.nonzero(~torch.isinf(T_vals).ravel()).ravel() # (B,)
    T_vals_valid = T_vals[bouncing_ray_indices].cpu() # (B x 1)
    valid_bouncing_ray_os = rays_o[bouncing_ray_indices].cpu() # (B x 3)
    valid_bouncing_ray_ds = rays_d[bouncing_ray_indices].cpu() # (B x 3)
    valid_bouncing_ray_ds = valid_bouncing_ray_ds * T_vals_valid # (B x 3)

    ax.quiver3D(valid_bouncing_ray_os[:, 0], valid_bouncing_ray_os[:, 1], valid_bouncing_ray_os[:, 2],
                valid_bouncing_ray_ds[:, 0], valid_bouncing_ray_ds[:, 1], valid_bouncing_ray_ds[:, 2],
                color='g', arrow_length_ratio=.05, linewidth=1.2, label='Camera Rays', alpha=.5)
    
    # Produce Legend.
    ax.legend()

    # Set views (https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html)
    azim_angles = np.linspace(0, 180, 8)
    for azim in azim_angles:
        # Set view
        ax.view_init(azim=azim)
        
        fig_name=f"combined_rayo_output_azim_{azim.item():.2f}.png"
        save_path = Path(__file__).parent / "graphs" / fig_name
        plt.savefig(save_path)
        print(f"Saved output figure at {str(save_path)}")


def plot_ray_o_and_d(ray_o: torch.Tensor, ray_d: torch.Tensor, render_fig = True, fig_name = "test_rayo_output.png", **quiver_kwargs):
    ray_o, ray_d = ray_o.cpu(), ray_d.cpu()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')


    ax.scatter(ray_o[:, 0], ray_o[:, 1], ray_o[:, 2], marker="o", c=ray_o[:, 2], cmap='tab10')
    ax.set_title(f"Generated Ray Origins and Directions")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.invert_yaxis()

    # Plot output normals:
    # https://matplotlib.org/stable/gallery/mplot3d/quiver3d.html
    ax.quiver3D(ray_o[:, 0], ray_o[:, 1], ray_o[:, 2], ray_d[:, 0], ray_d[:, 1], ray_d[:, 2], length = 0.5, cmap='tab10', arrow_length_ratio=.2, **quiver_kwargs)

    if render_fig:
        save_path = Path(__file__).parent / fig_name
        plt.savefig(save_path)
        print(f"Saved output figure at {str(save_path)}")

    return ax


def intersect_sphere(ray_o: torch.Tensor, ray_d: torch.Tensor, sphere_center: torch.Tensor, sphere_radius: float) -> torch.Tensor:
    """
    Intersect a group of rays with a sphere, returning the t_min of each ray that hits it (and the normal)

    ray_o: n x 3
    ray_d: n x 3
    sphere_center: 1 x 3
    sphere_radius: float

    TODO: Could also make into a slang kernel.
    """
    print(f"{ray_o.shape = }")
    print(f"{ray_d.shape = }")

    # Following https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection.html


    # Initial test to see if sphere is in same direction as ray
    L = sphere_center - ray_o
    T_center = col_wise_dot_product(ray_d, L) # (N,)

    # print(T_center < 0)
    # print(torch.sum(T_center < 0))

    # Distance from ray to sphere center
    L_distances = torch.linalg.norm(L, dim=1) # (N,)
    D_squared = L_distances**2 - T_center**2 # (N,)

    sphere_rad_squared = sphere_radius**2
    # print(D_squared < sphere_rad_squared)
    # print(torch.sum(D_squared < sphere_rad_squared))

    # T values
    T_hc = torch.sqrt(sphere_rad_squared - D_squared)
    T_zero = T_center - T_hc
    T_one = T_center + T_hc

    # T vals greater than 0 are only considered (filled with inf for comparison otherwise)
    t_min = 0
    T_zero = torch.where(T_zero < t_min, float("inf"), T_zero)
    T_one = torch.where(T_one < t_min, float("inf"),  T_one)

    # Need to be careful about NaNs here (even before minimum computation)
    # torch.nan_to_num?
    T_vals = torch.minimum(T_zero, T_one)
    T_vals = torch.nan_to_num(T_vals, nan=float("inf"), posinf=float("inf"))

    return T_vals.reshape(-1, 1) 

def col_wise_dot_product(arr_1, arr_2, keepdim=False):
    return torch.sum(arr_1 * arr_2, dim=1, keepdim=keepdim)# (N x 1)

def bounce_off_sphere(ray_o: torch.Tensor, ray_d: torch.Tensor, t_vals: torch.Tensor, sphere_center: torch.Tensor):
    """
    Returns bounce_ray_o and bounce_ray_d's for intersections with spheres with incoming rays at the given t_vals.

    ray_o: n x 3
    ray_d: n x 3
    ray_d: n x 3
    t_vals: n x 1
    sphere_center: 1 x 3

    TODO: Could also be a slang kernel.
    """
    # Initial hit position (bounce ray origin)
    hit_pos = ray_o + ray_d * t_vals # n x 3

    # Normalize sphere normals
    hit_normal_unnorm = hit_pos - sphere_center # n x 3
    hit_normal = hit_normal_unnorm / torch.linalg.norm(hit_normal_unnorm, dim=1, keepdim=True)

    # Reflect rays (bounce ray direction)
    reflected_rays = ray_d - 2 * hit_normal * (col_wise_dot_product(ray_d, hit_normal, keepdim=True))

    return hit_pos, reflected_rays



def get_rendering_cam(model_params: ModelParams, camera_index: int) -> MiniCam:
    camera_file = Path(model_params.model_path) / "cameras.json"
    with open(camera_file) as f:
        camera_arr = json.load(f)
        # print(f"Cameras: {camera_arr}")
        # print(f"Cameras[0]: {camera_arr[0]}")
        chosen_cam = camera_arr[camera_index]

        # Some more intermediate values:
        # Rt = np.zeros((4, 4))
        # Rt[:3, :3] = camera.R.transpose()
        # Rt[:3, 3] = camera.T
        # Rt[3, 3] = 1.0

        # W2C = np.linalg.inv(Rt)
        # pos = W2C[:3, 3]
        # rot = W2C[:3, :3]

        world_to_camera = torch.zeros((4, 4))
        world_to_camera[:3, 3] = torch.tensor(chosen_cam['position'])
        world_to_camera[:3, :3] = torch.tensor(chosen_cam['rotation'])
        world_to_camera[3, 3] = 1.0

        fovy = focal2fov(chosen_cam['fy'], chosen_cam['height'])
        fovx = focal2fov(chosen_cam['fx'], chosen_cam['width'])

        world_view_transform = world_to_camera.transpose(0, 1).to(device="cuda")
        z_near = 0.01
        z_far = 100
        proj_matrix = getProjectionMatrix(znear=z_near, zfar=z_far, fovX=fovx, fovY=fovy).transpose(0,1).to(device="cuda")
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(proj_matrix.unsqueeze(0))).squeeze(0)

        rendering_cam = MiniCam(chosen_cam['width'], chosen_cam['height'], fovy, fovx, 
                                0.01, 100, world_view_transform, full_proj_transform)
        
        rendering_cam.model = ProjectionType.PERSPECTIVE
                                
    return rendering_cam




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Manual Renderer Parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # args.checkpoint_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load Gaussians
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    gaussians = load_gaussian_model(lp.extract(args), op.extract(args), args.start_checkpoint)

    print(f"Loaded Gaussian, Active SH Degree: {gaussians.active_sh_degree}")

    render_gaussian_model(gaussians,lp.extract(args), op.extract(args), pp.extract(args))

    # All done
