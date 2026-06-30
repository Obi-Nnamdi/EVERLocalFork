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


def convert_pytorch_image_to_matplotlib(image: torch.Tensor) -> torch.Tensor:
    """
    (N, C, H, W) -> (H, W, C)
    """
    return image[0].permute(1, 2, 0).cpu()


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
    parser.add_argument("--path_to_model", "-p", required=True, type=str, default=None)
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
    brdf_normal_model = BRDF_normal_predictor(global_image_height, global_image_width)
    brdf_normal_model = brdf_normal_model.cuda()

    # Load model from checkpoint
    model_checkpoint_path = Path(args.path_to_model)
    print(f"Loading checkpoint at {model_checkpoint_path.absolute()}")

    model_state_dict = torch.load(model_checkpoint_path)
    brdf_normal_model.load_state_dict(model_state_dict)

    print(f"Loaded model checkpoint.")

    # Create a figure showing predicted diffuse, specular, and normal maps:
    camera_index = 128
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

    # Evaluate model for each point
    diffuse_coeffs = []
    spec_coeffs = []
    normals = []

    brdf_normal_model.eval()

    print(f"Running Model")
    start_time = time.process_time()
    with torch.no_grad():
        model_output = brdf_normal_model(rendered_image.unsqueeze(0))

    print(f"Ran model in {time.process_time() - start_time:.2f}s")
    print(f"{model_output = }")

    # Now plot the results using our points and outputs

    fig = plt.figure()
    fig.set_size_inches(8, 6)

    # First plot the rendered image that we're trying to match:
    ax = plt.subplot(2, 2, 1)
    ax.set_title("Main Image")

    rgb_image = rendered_image[:3, :, :].permute(1, 2, 0).clip(min = 0, max = 1)  # (H, W, C)
    ax.imshow(rgb_image.cpu())

    # Now the diffuse image (assume 0 where there's no information)
    ax = plt.subplot(2, 2, 2)
    ax.set_title("Diffuse Map")

    diffuse_image = convert_pytorch_image_to_matplotlib(
        model_output["brdf"]["diffuse"]
    ).clip(
        min=0, max=1
    )  # (H, W, C)
    ax.imshow(diffuse_image.cpu())

    # Same operation for specular
    ax = plt.subplot(2, 2, 3)
    ax.set_title("Specular Map")

    specular_image = convert_pytorch_image_to_matplotlib(
        model_output["brdf"]["specular"]
    ).clip(
        min=0, max=1
    )  # (H, W, C)
    ax.imshow(specular_image.cpu())

    # Show normals as normalized r, g, b images in world space (where the range is transformed from (-1, 1) to (0, 1))
    ax = plt.subplot(2, 2, 4)
    ax.set_title("Normal Map")

    normal_image = convert_pytorch_image_to_matplotlib(
        model_output["normal"]
    ).cuda()  # (H, W, C)

    camera_normals = nn.functional.normalize(normal_image, dim=-1)
    camera_normal_colors = (camera_normals / 2) + 0.5  # (-1, 1) -> (0, 1)

    # Calculate world normals (not displayed atm)
    camera_normals = camera_normals.reshape(-1, 3)
    world_normals = transform_normals_to_world_space(camera_normals, rendering_cam) # (N, 3)
    world_normal_colors = (world_normals / 2) + 0.5 # (-1, 1) -> (0, 1)

    world_normal_colors = world_normal_colors.reshape(
        global_image_height, global_image_width, 3
    )

    ax.imshow(camera_normal_colors.cpu())

    # Save figure out
    plt.suptitle(f"Plots of Diffuse and Specular Maps for Camera {camera_index}")
    save_path = Path(model_params.model_path) / f"model_diffuse_specular.png"
    plt.savefig(save_path, dpi=300)

    print(f"Saved figure at {save_path.absolute()}")
    # All done
