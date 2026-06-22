from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser
from neural_brdf import (
    BRDF_normal_predictor,
    transform_normals_to_world_space,
)
from raytracing import (
    build_gaussian_renderer,
    depth_map_to_xyz,
    gather_incoming_light_at_point,
    get_cameras,
    get_rendering_cam,
    load_gaussian_model,
    render_gaussians,
    generate_spherical_rays,
    gather_incoming_light_at_points,
)
from utils.general_utils import safe_state
import sys
from tqdm import tqdm

import torch
from torch import nn

from pathlib import Path
import os

import time

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Manual Renderer Parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # args.checkpoint_iterations.append(args.iterations)

    print("Reading from " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load Gaussians
    model_params: ModelParams = lp.extract(args)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    gaussians = load_gaussian_model(
        model_params, op.extract(args), args.start_checkpoint
    )
    print(f"Loaded Gaussian, Active SH Degree: {gaussians.active_sh_degree}")

    rendering_cameras = get_cameras(model_params)
    num_cameras = len(rendering_cameras)

    # Set a global image width and height that is used for instanciating the neural network, etc.
    preview_factor = 4
    global_image_height = rendering_cameras[0].image_height // preview_factor
    global_image_width = rendering_cameras[0].image_width // preview_factor

    # Set up our initial renderer
    ever_renderer = build_gaussian_renderer(
        gaussians, rendering_cameras[0], pp.extract(args)
    )

    # More constants (affecting how much incoming light we use)
    incoming_light_sphere_divisions = 4

    # Calculate how big our incoming light features will be when input into our model.
    test_sphere_o, _ = generate_spherical_rays(
        torch.zeros((3,)), incoming_light_sphere_divisions
    )
    incoming_light_size = test_sphere_o.size(0) * 3  # N vectors that have [r, g, b]

    # Instanciate the BRDF_normal_predictor
    brdf_normal_model = BRDF_normal_predictor(
        global_image_height, global_image_width, incoming_light_size
    )
    brdf_normal_model = brdf_normal_model.cuda()

    # Load model from checkpoint
    model_checkpoint_dir = "brdf_models"
    model_checkpoint_path = Path(model_params.model_path) / model_checkpoint_dir / "brdf_model.pt"
    print(f"Loading checkpoint at {model_checkpoint_path.absolute()}")

    model_state_dict = torch.load(model_checkpoint_path)
    brdf_normal_model.load_state_dict(model_state_dict)
    
    print(f"Loaded model checkpoint.")

    # Create a figure showing predicted diffuse, specular, and normal maps:
    camera_index = 1
    rendering_cam = rendering_cameras[camera_index]
    # print(f"{global_image_width = }")
    # print(f"{global_image_height = }")
    rendering_cam.image_width = global_image_width
    rendering_cam.image_height = global_image_height

    rendered_image = render_gaussians(
            ever_renderer, rendering_cam, None, include_depth=True
    )  # (C, H, W)

    # Separate RGB and Depth Images
    depth_map = rendered_image[3, :, :]  # (H, W)

    # TODO: Iterate over all points?
    rand_row = torch.randint(0, global_image_height, (1,)).item()
    rand_col = torch.randint(0, global_image_width, (1,)).item()

    # Get Incoming Light for all points
    rays_o, rays_d = ever_renderer.get_rays(rendering_cam)
    xyz_map = depth_map_to_xyz(rays_o, rays_d, depth_map)  # (H, W, 3)
    
    # all_rows, all_cols = torch.meshgrid([torch.arange(global_image_height), torch.arange(global_image_width)])
    all_rows, all_cols = torch.meshgrid([torch.arange(100, 150), torch.arange(100, 150)])
    
    all_rows = all_rows.ravel()
    all_cols = all_cols.ravel()

    print(f"{all_rows.shape = }")
    print(f"{xyz_map = }")
    print(f"{xyz_map[(all_rows, all_cols)] = }")

    print("Calculating Incoming Light")
    start_time = time.process_time()
    # Querying Spherical Directions
    incoming_light, _, incoming_light_dirs = (
        gather_incoming_light_at_points(
            xyz_map[(all_rows, all_cols)],
            ever_renderer,
            tmin=0.01,
            sphere_divisions=incoming_light_sphere_divisions,
            fast=True
        )
    ) # (N, R, 3) for each tensor
    print(f"Calculated Incoming Light in {time.process_time() - start_time:.2f}s")

    # Evaluate model for each point
    diffuse_coeffs = []
    spec_coeffs = []
    normals = []

    # How many points are we working with?
    N = all_rows.size(0)

    brdf_normal_model.eval()

    print(f"Running Model")
    start_time = time.process_time()
    with torch.no_grad():
        # Create a block of rendered images that are just the same image duplicated for each point
        input_images = rendered_image.unsqueeze(0).expand(N, -1, -1, -1) # (N, C, H, W)
        model_output = brdf_normal_model(input_images, incoming_light)

    print(f"Ran model in {time.process_time() - start_time:.2f}s")
    print(f"{model_output = }")

    # Now plot the results using our points and outputs

    fig = plt.figure()
    fig.set_size_inches(5, 12)

    # First plot the rendered image that we're trying to match:
    ax = plt.subplot(4, 1, 1) 
    ax.set_title("Main Image")

    rgb_image = rendered_image[:3, :, :].permute(1, 2, 0).clip(min = 0, max = 1)  # (H, W, C)
    ax.imshow(rgb_image.cpu())

    # Now the diffuse image (assume 0 where there's no information)
    ax = plt.subplot(4, 1, 2) 
    ax.set_title("Diffuse Map")

    diffuse_image = torch.zeros_like(rgb_image) # (H, W, C)
    diffuse_image[all_rows, all_cols] = model_output["brdf"]["diffuse"] # both (N, 3) so it works
    diffuse_image = diffuse_image.clip(min = 0, max = 1)
    ax.imshow(diffuse_image.cpu())

    # Same operation for specular
    ax = plt.subplot(4, 1, 3) 
    ax.set_title("Specular Map")

    specular_image = torch.zeros_like(rgb_image) # (H, W, C)
    specular_image[all_rows, all_cols] = model_output["brdf"]["specular"] # both (N, 3) so it works
    specular_image = specular_image.clip(min = 0, max = 1)
    ax.imshow(specular_image.cpu())

    # Show normals as normalized r, g, b images in world space (where the range is transformed from (-1, 1) to (0, 1))
    ax = plt.subplot(4, 1, 4) 
    ax.set_title("Normal Map")

    normal_image = torch.zeros_like(rgb_image) # (H, W, C)

    camera_normals = nn.functional.normalize(model_output["normal"])
    world_normals = transform_normals_to_world_space(camera_normals, rendering_cam) # (N, 3)
    world_normal_colors = (world_normals / 2) + 0.5 # (-1, 1) -> (0, 1)

    normal_image[all_rows, all_cols] = world_normal_colors # both (N, 3)
    ax.imshow(normal_image.cpu())

    # Save figure out
    plt.suptitle(f"Plots of Diffuse and Specular Maps for Camera {camera_index}")
    save_path = Path(model_params.model_path) / f"model_diffuse_specular.png"
    plt.savefig(save_path, dpi=300)

    print(f"Saved figure at {save_path.absolute()}")
    # All done
