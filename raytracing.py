from pathlib import Path

from arguments import ModelParams, OptimizationParams
from scene import GaussianModel
from scene.dataset_readers import ProjectionType
from utils.graphics_utils import focal2fov, getProjectionMatrix
from utils.system_utils import searchForMaxIteration
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser
import json
import cv2
from utils.general_utils import safe_state
from typing import Optional
import sys


from gaussian_renderer.fast_renderer import FastRenderer

from scene.cameras import MiniCam

import torch


import os
import numpy as np

# Graphing
import matplotlib

matplotlib.use("Agg")  # headless mode
import matplotlib.pyplot as plt

def load_gaussian_model(
    model_params: ModelParams,
    opt_params: OptimizationParams,
    checkpoint: os.PathLike | None,
) -> GaussianModel:
    first_iter = 0
    gaussians = GaussianModel(
        model_params.sh_degree,
        model_params.use_neural_network,
        model_params.max_opacity,
    )

    if checkpoint is not None:
        gaussian_params, first_iter = torch.load(checkpoint)
        gaussians.restore(gaussian_params, opt_params)

    # Search for a valid EVER-comaptible .ply
    else:
        loaded_iter = searchForMaxIteration(
            os.path.join(model_params.model_path, "point_cloud")
        )
        print("Loading trained model at iteration {}".format(loaded_iter))

        gaussians.load_ply(
            os.path.join(
                model_params.model_path,
                "point_cloud",
                "iteration_" + str(loaded_iter),
                "point_cloud.ply",
            )
        )

    return gaussians


def build_gaussian_renderer(
    gaussians: GaussianModel, camera: MiniCam, pipe_params: PipelineParams
) -> FastRenderer:
    renderer = FastRenderer(camera, gaussians, pipe_params.enable_GLO)
    renderer.set_camera(camera)

    # TODO: Maybe set a custom height/width?

    return renderer


def render_gaussians(
    renderer: FastRenderer,
    camera: MiniCam,
    tmin: Optional[float] = None,
    include_depth=True,
) -> torch.Tensor:
    """
    Returns a (channels x H x W) image.
    """
    # gaussians.training_setup(opt_params)
    # torch.cuda.empty_cache()

    # "None" arguments aren't used in render function.
    # TODO: Should be refactored.
    return renderer.render(camera, None, None, tmin, include_depth=include_depth)


def depth_map_to_xyz(
    rays_o: torch.Tensor, rays_d: torch.Tensor, depth_map: torch.Tensor
) -> torch.Tensor:
    """
    Convert rendered depth z-depth to xyz points (assuming everything in world space)
    rays_o: ((h*w) x 3)
    rays_d: ((h*w) x 3)
    depth_map: (h x w x 1)

    Returns: (h x w x 3)
    """

    depth_map_unrolled = depth_map.reshape(-1, 1)  # (h*w, 1)
    xyz_points = rays_o + rays_d * depth_map_unrolled

    return xyz_points.reshape(depth_map.shape[0], depth_map.shape[1], 3)


def gather_incoming_light_at_point(
    point: torch.Tensor, renderer: FastRenderer, tmin=0.01
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Spherical Queries for incoming light.

    Color returned as an N x 3 tensor.
    """
    # torch.cuda.empty_cache()
    # Make sphere ray directions (https://matplotlib.org/stable/gallery/mplot3d/surface3d_2.html)
    # TODO: Research better method
    # TODO: Should this be spherical or hemispherical?
    # TODO: Do I need to check for unique points?

    # Use a sphere centered at the origin to build the ray directions as a hemisphere query
    radius = 1
    probe_divisions = 10
    u = torch.linspace(0, 2 * torch.pi, probe_divisions)
    v = torch.linspace(0, torch.pi, probe_divisions)
    x = radius * torch.outer(torch.cos(u), torch.sin(v)).reshape(-1, 1)
    y = radius * torch.outer(torch.sin(u), torch.sin(v)).reshape(-1, 1)
    z = radius * torch.outer(torch.ones_like(u), torch.cos(v)).ravel().reshape(-1, 1)

    # print(f"{x.shape = }")
    # print(f"{x = }")

    # No normalization needed
    rays_d = torch.hstack((x, y, z)).to(device="cuda")

    rays_o = point.to(device="cuda")
    rays_o = rays_o.expand(rays_d.shape).contiguous()

    probe_image = renderer.trace_rays_from_single_rayo(rays_o, rays_d, tmin, 1e7)

    # (N x 3) tensor since there's not much of an "image" here to coerce to 2D.
    return probe_image["color"][:, :3], rays_o, rays_d


def save_rgb_image(
    image: torch.Tensor, img_path: Path, resized_dims: Optional[tuple[int, int]] = None
):
    """
    Saves an RGB or RGBA image in the format (channels, height, width).
    """
    converted_img = (
        (torch.clamp(image, min=0, max=1.0) * 255)
        .byte()
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )
    if resized_dims is not None:
        image_width, image_height = resized_dims
        converted_img = cv2.resize(image, (image_width, image_height))

    converted_img = cv2.cvtColor(converted_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(
        img_path,
        converted_img,
    )

    return converted_img


def get_rendering_cam(model_params: ModelParams, camera_index: int) -> MiniCam:
    """
    Get the parameters of a camera from a cameras.json file at the specified index.
    """
    camera_file = Path(model_params.model_path) / "cameras.json"
    with open(camera_file) as f:
        camera_arr = json.load(f)
        chosen_cam = camera_arr[camera_index]

        # TODO: Fixed values for rotation but for some reason the camera position is still inaccurate.
        # Investigate eventually (camera_utils.py).

        world_to_camera = torch.zeros((4, 4))
        world_to_camera[3, :3] = torch.tensor(chosen_cam["position"])
        world_to_camera[:3, :3] = torch.tensor(chosen_cam["rotation"])
        world_to_camera[3, 3] = 1.0

        # Closer match to camera 1 (From manually inspecting Camera values from host_render_server)
        hard_coded_w2c = torch.tensor(
            [
                [-9.9864e-01, -1.5685e-03, 5.2073e-02, 0.0000e00],
                [-2.6808e-02, 8.7253e-01, -4.8783e-01, 0.0000e00],
                [-4.4670e-02, -4.8856e-01, -8.7139e-01, -0.0000e00],
                [1.2334e-01, 5.3778e-02, 3.2802e00, 1.0000e00],
            ]
        )

        fovy = focal2fov(chosen_cam["fy"], chosen_cam["height"])
        fovx = focal2fov(chosen_cam["fx"], chosen_cam["width"])

        world_view_transform = world_to_camera.to(device="cuda")
        z_near = 0.01
        z_far = 1000
        proj_matrix = (
            getProjectionMatrix(znear=z_near, zfar=z_far, fovX=fovx, fovY=fovy)
            .transpose(0, 1)
            .to(device="cuda")
        )
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(proj_matrix.unsqueeze(0))
        ).squeeze(0)

        rendering_cam = MiniCam(
            chosen_cam["width"],
            chosen_cam["height"],
            fovy,
            fovx,
            z_near,
            z_far,
            world_view_transform,
            full_proj_transform,
        )

        rendering_cam.model = ProjectionType.PERSPECTIVE

    return rendering_cam


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

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load Gaussians
    model_params: ModelParams = lp.extract(args)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    gaussians = load_gaussian_model(
        model_params, op.extract(args), args.start_checkpoint
    )
    print(f"Loaded Gaussian, Active SH Degree: {gaussians.active_sh_degree}")

    camera_index = 1
    rendering_cam = get_rendering_cam(model_params, camera_index)

    # Reduce Camera resolution
    preview_factor = 4
    image_width = rendering_cam.image_width
    image_height = rendering_cam.image_height

    rendering_cam.image_width = image_width // preview_factor
    rendering_cam.image_height = image_height // preview_factor

    # Build Renderer and Render
    renderer = build_gaussian_renderer(gaussians, rendering_cam, pp.extract(args))
    rendered_image = render_gaussians(renderer, rendering_cam, None)

    save_rgb_image(rendered_image, Path(model_params.model_path) / "camera_1_img.png")
    print(f"Saved image.")

    # Depth map to XYZ Pipeline
    depth_map = rendered_image[3, :, :]
    # TODO: How to handle any t_max occurences? Will have to look into that
    # since this alone doesn't work (maybe a threshold?).
    # print("T_MAX Depth Map:")
    # print(torch.sum(depth_map == 1e7))

    rays_o, rays_d = renderer.get_rays(rendering_cam)
    xyz_map = depth_map_to_xyz(rays_o, rays_d, depth_map)

    chosen_point = (742, 416)  # Should be on a flower
    print(f"{xyz_map[chosen_point] = }")

    # Querying Spherical Directions
    incoming_light, rays_o, sphere_rays_d = gather_incoming_light_at_point(
        xyz_map[chosen_point], renderer, tmin=0.01
    )

    # TODO: Plot this
    sphere_rays_d = sphere_rays_d.cpu()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title(f"Incoming Light for Camera {camera_index} at point {chosen_point}")

    clamped_light = torch.clamp(incoming_light, min=0, max=1.0).cpu()
    ax.scatter(
        sphere_rays_d[:, 0], sphere_rays_d[:, 1], sphere_rays_d[:, 2], c=clamped_light
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    save_path = Path(model_params.model_path) / f"incoming_light_{chosen_point}.png"
    plt.savefig(save_path)
    print(f"Saved output figure at {str(save_path)}")

    # Save output for verification in blender (radius of 1)
    xyz_coords = sphere_rays_d + rays_o.cpu()
    xyz_color_coords = torch.hstack((xyz_coords, incoming_light.cpu()))  #  (N x 6)

    np.save(
        Path(model_params.model_path)
        / f"incoming_light_probe_cam_{camera_index}_point_{chosen_point}.npy",
        xyz_color_coords.numpy(),
    )

    # TODO: Improve speed (i.e. do this for every single point in parallel)

    # All done
