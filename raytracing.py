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
from matplotlib.axes import Axes

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
    prerender_shs=True,
) -> torch.Tensor:
    """
    Returns a (channels x H x W) image.
    """
    # gaussians.training_setup(opt_params)
    # torch.cuda.empty_cache()

    # "None" arguments aren't used in render function.
    # TODO: Should be refactored.
    renderer.set_camera(camera)
    return renderer.render(
        camera,
        None,
        None,
        tmin,
        include_depth=include_depth,
        prerender_shs=prerender_shs,
    )


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
    point: torch.Tensor,
    renderer: FastRenderer,
    tmin=0.01,
    sphere_divisions=10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Spherical Queries for incoming light.

    Color, Ray Origins, and Ray directions returned as N x 3 tensors.
    """

    # torch.cuda.empty_cache()
    # TODO: Research better method
    # TODO: Should this be spherical or hemispherical? (spherical for now, hemispherical would be best w/ normals)

    rays_d, rays_o = generate_spherical_rays(point, divisions=sphere_divisions)
    probe_image = renderer.trace_rays_from_single_rayo(rays_o, rays_d, tmin, 1e7)

    # (N x 3) tensor since there's not much of an "image" here to coerce to 2D.
    return probe_image["color"][:, :3], rays_o, rays_d


def gather_incoming_light_at_points(
    points: torch.Tensor,
    renderer: FastRenderer,
    tmin=0.01,
    sphere_divisions=10,
    fast=True,
    precompute_sh=False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Inputs:
        points: (N, 3) of all the points to get light from
        renderer: Renderer to query light directions from
        tmin: Starting t value that rays cast from
        sphere_divisions: divisions to break up light queries into

    Outputs:
        incoming_light: (N, R, 3) where each point gets R sources of incoming light
        sphere_ray_o: (N, R, 3)
        sphere_ray_d: (N, R, 3)
    """
    base_rays_d, _ = generate_spherical_rays(
        torch.zeros(
            3,
        ),
        divisions=sphere_divisions,
    )  # (R, 3), (R, 3)

    R = base_rays_d.size(0)
    N = points.size(0)

    base_rays_o = points.unsqueeze(1).expand(-1, R, -1).contiguous()  # (N, R, 3)
    base_rays_d = base_rays_d.unsqueeze(0).expand(N, -1, -1).contiguous()  # (N, R, 3)

    if fast:
        # One Big call to trace_rays
        rays_o = base_rays_o.reshape(N * R, 3)
        rays_d = base_rays_d.reshape(N * R, 3)

        # TODO: Using trace_rays_from_single_rayo is much more accurate to the slow method than this
        # A little concerned about that fact, but continuing to use this method for now.
        if precompute_sh:
            probe_image = renderer.trace_rays_from_single_rayo(
                rays_o, rays_d, tmin, 1e7
            )

        else:
            probe_image = renderer.trace_rays_using_shs(rays_o, rays_d, tmin, 1e7)

        colors = probe_image["color"][:, :3]  # (N * R, 3)
        incoming_light = colors.reshape(N, R, 3)

        # TODO: Use returned t values to filter for directions that didn't intersect an ellipsoid,
        # and color it according to an environment map (or maybe always have an environment map in the background)
        # t_values = probe_image["saved"].states[:, 7]  # (N * R)
        # print(
        #     f"{colors[t_values == 0,: ] = }"
        # )  # (0,0,0) colors, so rays that didn't hit anything.
        # print(f"{probe_image['saved'].states[:, 12][t_values == 0] = }")  # logT values (also 0)

    else:
        # Manually precompute spherical harmonics for each ray group
        # TODO: Largely unnneeded now except for possible memory concerns.
        colors = []

        for i in range(N):
            rays_o = base_rays_o[i, :, :]
            rays_d = base_rays_d[i, :, :]

            if precompute_sh:
                probe_image = renderer.trace_rays_from_single_rayo(
                    rays_o, rays_d, tmin, 1e7
                )
            else:

                probe_image = renderer.trace_rays_using_shs(rays_o, rays_d, tmin, 1e7)
            colors.append(probe_image["color"][:, :3])  # (R, 3)

        incoming_light = torch.stack(colors, dim=0)  # (N, R, 3)

    return incoming_light, base_rays_o, base_rays_d


def generate_spherical_rays(
    point: torch.Tensor, divisions=10
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Use a sphere centered at the origin to build the ray directions as a hemisphere query
    Parametric Eq for sphere (https://math.stackexchange.com/questions/150937/derive-parametric-equations-for-sphere):
    x = rcos(theta)sin(phi)
    y = rsin(theta)sin(phi)
    z = rcos(theta)
    theta from 0 -> 2pi, phi from 0 -> pi

    Code adapted from https://matplotlib.org/stable/gallery/mplot3d/surface3d_2.html.

    Output:
        rays_o: (N, 3)
        rays_d: (N, 3)
    """

    radius = 1
    u = torch.linspace(0, 2 * torch.pi, divisions)
    v = torch.linspace(0, torch.pi, divisions)

    clamp_to_zero = lambda tensor: torch.where(torch.abs(tensor) < 1e-6, 0, tensor)

    # Clamp small values to 0s (1s are fine from inspection)
    sin_u = clamp_to_zero(torch.sin(u))
    sin_v = clamp_to_zero(torch.sin(v))
    cos_u = clamp_to_zero(torch.cos(u))
    cos_v = clamp_to_zero(torch.cos(v))
    x = radius * torch.outer(cos_u, sin_v).reshape(-1, 1)
    y = radius * torch.outer(sin_u, sin_v).reshape(-1, 1)
    z = radius * torch.outer(torch.ones_like(u), cos_v).reshape(-1, 1)

    # No normalization needed
    rays_d = torch.hstack((x, y, z))

    # Get unique directions
    # TODO: Might be nice to look into a more foolproof way to do this
    rays_d = torch.unique(rays_d, dim=0)
    rays_d = rays_d.cuda()

    rays_o = point.to(device="cuda")
    rays_o = rays_o.expand(rays_d.shape).contiguous()
    return rays_d, rays_o


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

        rendering_cam = extract_cam_from_json(chosen_cam)

    return rendering_cam


def extract_cam_from_json(json_cam: dict) -> MiniCam:
    """
    Extracts a `MiniCam` object from a single entry from a cameras.json file.
    """
    rotation_matrix = torch.tensor(json_cam["rotation"])

    world_to_camera = torch.zeros((4, 4))
    # W2C Rotation
    world_to_camera[:3, :3] = rotation_matrix

    # Change coordinate frame of position properly by using rotation matrix and inverting (take negative)
    world_to_camera[3, :3] = -(
        torch.linalg.inv(rotation_matrix) @ torch.torch.tensor(json_cam["position"])
    )
    world_to_camera[3, 3] = 1.0

    fovy = focal2fov(json_cam["fy"], json_cam["height"])
    fovx = focal2fov(json_cam["fx"], json_cam["width"])

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
        json_cam["width"],
        json_cam["height"],
        fovy,
        fovx,
        z_near,
        z_far,
        world_view_transform,
        full_proj_transform,
    )

    rendering_cam.model = ProjectionType.PERSPECTIVE
    return rendering_cam


def get_cameras(model_params: ModelParams) -> list[MiniCam]:
    # Get number of cameras:
    camera_file = Path(model_params.model_path) / "cameras.json"
    with open(camera_file) as f:
        camera_arr = json.load(f)

    return [extract_cam_from_json(camera) for camera in camera_arr]


def plot_incoming_light_and_outgoing_radiance(
    incoming_light_colors: torch.Tensor,
    incoming_light_locations: torch.Tensor,
    outgoing_radiance: torch.Tensor,
    outgoing_dir: torch.Tensor,
) -> Axes:
    """
    Plot the incoming light and outgoing radiance at a point.
    Inputs:
        incoming_light: (N, 3)
        outgoing_radiance: (1, 3)
        outgoing_dir: (1, 3)
        incoming_light_locations: (N, 3)
    """
    incoming_light_locations = incoming_light_locations.detach().cpu()
    incoming_light_colors = incoming_light_colors.detach().cpu().clip(0, 1)
    outgoing_dir = outgoing_dir.detach().cpu()
    outgoing_radiance = outgoing_radiance.detach().cpu().clip(0, 1)

    ax = plt.gca()
    ax.scatter(
        incoming_light_locations[:, 0],
        incoming_light_locations[:, 1],
        incoming_light_locations[:, 2],
        c=incoming_light_colors,
    )
    ax.quiver3D(
        [0],
        [0],
        [0],
        outgoing_dir[:, 0],
        outgoing_dir[:, 1],
        outgoing_dir[:, 2],
        color="g",
        arrow_length_ratio=0.3,
        linewidth=3,
        label="Outgoing Direction",
        alpha=1,
    )

    # Put the outgoing radiance in the center and make it larger
    ax.scatter([0], [0], 0, c=outgoing_radiance, s=150)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return ax


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
    rendered_image_sh = render_gaussians(
        renderer, rendering_cam, None, prerender_shs=False
    )
    print("Difference between pre-computing SHs and not:")
    print(f"{rendered_image[:3] - rendered_image_sh[:3] =}")
    print(
        f"{torch.nn.functional.mse_loss(rendered_image[:3], rendered_image_sh[:3]) =}"
    )

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

    chosen_point = (145, 511)  # Should be on the very top flower

    # Testing light querying from multiple chosen points
    print(f"Testing Multiple Points of Incoming Light")
    torch.set_printoptions(sci_mode=False)

    num_points = 6
    sphere_divisions = 4
    rand_row = torch.randint(0, rendering_cam.image_height, (num_points,))
    rand_col = torch.randint(0, rendering_cam.image_width, (num_points,))
    # random_points = (torch.rand((5, 3)) * 40).cuda()
    # random_points = xyz_map[(rand_row, rand_col)][3:8, :]
    random_points = xyz_map[(rand_row, rand_col)]
    incoming_light_fast, all_rays_o, all_rays_d = gather_incoming_light_at_points(
        random_points,
        renderer,
        sphere_divisions=sphere_divisions,
        fast=True,
        precompute_sh=False,
    )
    torch.cuda.synchronize()

    # print(f"{all_rays_o = }")
    # print(f"{all_rays_d = }")
    incoming_light_fast_inacc, _, _ = gather_incoming_light_at_points(
        random_points,
        renderer,
        sphere_divisions=sphere_divisions,
        fast=True,
        precompute_sh=True,
    )
    incoming_light_slow, _, _ = gather_incoming_light_at_points(
        random_points,
        renderer,
        sphere_divisions=sphere_divisions,
        fast=False,
        precompute_sh=False,
    )
    incoming_light_slow_inacc, _, _ = gather_incoming_light_at_points(
        random_points,
        renderer,
        sphere_divisions=sphere_divisions,
        fast=False,
        precompute_sh=True,
    )

    print(f"{incoming_light_fast = }")
    print(f"{incoming_light_fast_inacc = }")
    print(f"{incoming_light_slow = }")
    print(f"{incoming_light_slow_inacc = }")
    print(f"{incoming_light_slow - incoming_light_fast_inacc = }")
    print(f"{incoming_light_slow - incoming_light_fast = }")
    print(f"{incoming_light_slow_inacc - incoming_light_fast_inacc = }")

    print(
        f"{torch.nn.functional.mse_loss(incoming_light_fast, incoming_light_slow) = }"
    )
    print(
        f"{torch.nn.functional.mse_loss(incoming_light_fast_inacc, incoming_light_slow) = }"
    )

    # Querying Spherical Directions
    incoming_light, rays_o, sphere_rays_d = gather_incoming_light_at_point(
        xyz_map[chosen_point], renderer, tmin=0.01
    )

    # Plotting incoming light
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
