from arguments import (
    ModelParams,
    PipelineParams,
    OptimizationParams,
    BRDFOptmizationParams,
)
from argparse import ArgumentParser
from raytracing import (
    build_gaussian_renderer,
    depth_map_to_xyz,
    get_cameras,
    load_gaussian_model,
    render_gaussians,
    generate_spherical_rays,
    gather_incoming_light_at_points,
)
from utils.general_utils import safe_state
from utils.tensor_utils import size_of_tensor_bytes
import sys
from tqdm import tqdm

import torch

from pathlib import Path
import os

from typing import cast, TypedDict

# Graphing
import matplotlib

matplotlib.use("Agg")  # headless mode


# Class for the returned cache dictionary
class BRDFCacheDict(TypedDict):
    full_rendered_images: torch.Tensor  # (N, C, H, W)
    incoming_light_probe_colors: torch.Tensor  # (P, R, 3)
    incoming_light_probe_directions: torch.Tensor  # (R, 3)
    incoming_light_probe_query: torch.Tensor  # (N, 1, H, W)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Manual Renderer Parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    brdf_optim_params = BRDFOptmizationParams(parser)
    parser.add_argument(
        "--start_ever_checkpoint",
        type=str,
        default=None,
        help="Checkpoint to resume ever model from.",
    )
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--caching_batch_size", type=int, default=1000, help="The batch size to use when computing the mapping from the incoming light probe to all camera images.")
    parser.add_argument(
        "--incoming_light_batch_size",
        type=int,
        default=200_000,
        help="The batch size to use when computing incoming light for the incoming light probe.",
    )
    parser.add_argument(
        "--num_probe_points",
        type=int,
        default=250_000,
        help="The number of points to use to create a probe for the incoming light",
    )
    args = parser.parse_args(sys.argv[1:])
    brdf_args = cast(
        BRDFOptmizationParams, brdf_optim_params.extract(args)
    )  # NOTE: Lying to the type checker, but it's close enough.

    print("Generating Cache for EVER Model at " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load Gaussians
    model_params: ModelParams = cast(ModelParams, lp.extract(args))

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    gaussians = load_gaussian_model(
        model_params,
        cast(OptimizationParams, op.extract(args)),
        args.start_ever_checkpoint,
    )
    print(f"Loaded Gaussian, Active SH Degree: {gaussians.active_sh_degree}")

    rendering_cameras = get_cameras(model_params)
    num_cameras = len(rendering_cameras)

    # Define our save path (inside the model_dir)
    cache_save_folder = Path(model_params._model_path) / "brdf_ever_cache"

    # Set a global image width and height that is used for generating our cache of incoming light
    global_image_height = cast(
        int, rendering_cameras[0].image_height // brdf_args.preview_factor
    )
    global_image_width = cast(
        int, rendering_cameras[0].image_width // brdf_args.preview_factor
    )

    print(
        f"Rendering images at a {global_image_width} x {global_image_height} resolution (W x H)."
    )

    # Set up our initial renderer
    ever_renderer = build_gaussian_renderer(
        gaussians, rendering_cameras[0], cast(PipelineParams, pp.extract(args))
    )

    # Calculate how big our incoming light features will be when input into our model.
    test_sphere_o, _ = generate_spherical_rays(
        torch.zeros((3,)), brdf_args.incoming_light_divisions
    )
    incoming_light_size = test_sphere_o.size(0)  # R dimension

    # Allocate space for all of our big tensors (on CPU for more RAM, we'll copy from GPU over on each iteration)
    # full_incoming_light_tensor = torch.zeros(
    #     num_cameras, global_image_height * global_image_width, incoming_light_size, 3, dtype=torch.float
    # )  # (N, P, R, 3)

    num_incoming_light_probe_points = cast(
        int, args.num_probe_points
    )  # (or H * W for consistency?)
    incoming_light_probe_tensor = torch.zeros(
        num_incoming_light_probe_points, incoming_light_size, 3, dtype=torch.float
    )  # (P, R, 3)

    # Tells each point where it needs to get its incoming light information from (I wish I could go down to
    #  int16 but the indices into the probe tensor= will likely be huge unless I guarantee it's smaller than 65K ish...)
    incoming_light_probe_query_tensor = torch.zeros(
        num_cameras,
        1,  # Index only
        global_image_height,
        global_image_width,
        dtype=torch.int32,
    )  # (N, 1, H, W), will get actually created later.

    # Not saved, but a giant point cloud (XYZ) of all of our points from all of our cameras.
    full_scene_point_cloud = torch.zeros(
        num_cameras, global_image_height * global_image_width, 3, dtype=torch.float
    )  # (N, H * W, 3)

    num_channels = 4  # (R, G, B, D)
    full_rendered_images_tensor = torch.zeros(
        num_cameras,
        num_channels,
        global_image_height,
        global_image_width,
        dtype=torch.float,
    )  # (N, C, H, W)

    print(f"Allocated Tensors.")
    # print(f"Full incoming light tensor size (GB): {full_incoming_light_tensor.element_size() * full_incoming_light_tensor.nelement() / 1024 / 1024 / 1024}")
    print(
        f"Full incoming light probe tensor size (GB): {size_of_tensor_bytes(incoming_light_probe_tensor) / 1024 / 1024 / 1024}"
    )
    print(
        f"Probe query tensor size (GB): {size_of_tensor_bytes(incoming_light_probe_query_tensor) / 1024 / 1024 / 1024}"
    )
    print(
        f"Full camera images tensor size (GB): {size_of_tensor_bytes(full_rendered_images_tensor) / 1024 / 1024 / 1024}"
    )
    print(
        f"Full camera point cloud tensor size (GB): {size_of_tensor_bytes(full_scene_point_cloud) / 1024 / 1024 / 1024}"
    )

    # Fill up these huge tensors index by index
    print(f"Performing initial fill of our camera rendering and XYZ tensors...")
    for camera_index in tqdm(range(num_cameras)):
        torch.cuda.empty_cache()

        rendering_cam = rendering_cameras[camera_index]

        rendering_cam.image_width = global_image_width
        rendering_cam.image_height = global_image_height

        rendered_image = render_gaussians(
            ever_renderer, rendering_cam, None, include_depth=True
        )  # (C, H, W)

        full_rendered_images_tensor[camera_index] = rendered_image

        depth_map = rendered_image[3, :, :]  # (H, W)

        # Make our xyz_map for each pixel
        rays_o, rays_d = ever_renderer.get_rays(rendering_cam)
        xyz_map = depth_map_to_xyz(rays_o, rays_d, depth_map)  # (H, W, 3)
        xyz_map = xyz_map.reshape(-1, 3)  # (H * W, 3)

        full_scene_point_cloud[camera_index] = xyz_map

    print("Downsampling Point Cloud...")
    # Now downsample our point cloud and get our incoming light for each of these points
    collapsed_point_cloud = full_scene_point_cloud.view(
        num_cameras * global_image_height * global_image_width, 3
    )  # (N * H * W, 3)

    # NOTE: Could also be done via a torch rand call and a topK. Multinom doesn't work (too many categories)
    try:
        rand_points = torch.randperm(
            num_cameras * global_image_height * global_image_width, device="cuda"
        )[:num_incoming_light_probe_points].cpu()
    except torch.OutOfMemoryError:
        print(f"Downsampling on CPU....")
        rand_points = torch.randperm(
            num_cameras * global_image_height * global_image_width
        )[:num_incoming_light_probe_points]

    probe_point_xyz = collapsed_point_cloud[rand_points, :]  # (P, 3)

    probe_point_xyz = probe_point_xyz.cuda()
    collapsed_point_cloud = collapsed_point_cloud.cuda()

    print("Generating Incoming Light Probe...")
    full_incoming_light_colors = torch.empty(
        0,
    ).cuda()

    # Generate our incoming light in batches according to our batch size:
    incoming_light_batch_size = cast(int, args.incoming_light_batch_size)
    probe_point_batches = torch.split(probe_point_xyz, incoming_light_batch_size, dim=0)
    incoming_light_dirs = None
    for probe_batch in tqdm(probe_point_batches, total=len(probe_point_batches)):
        incoming_light_colors, _, incoming_light_dirs = gather_incoming_light_at_points(
            probe_batch,
            ever_renderer,
            tmin=brdf_args.incoming_light_tmin,
            sphere_divisions=brdf_args.incoming_light_divisions,
            fast=True,
            precompute_sh=False,
        )  # (P, R, 3)
        full_incoming_light_colors = torch.cat(
            [full_incoming_light_colors, incoming_light_colors]
        )

    incoming_light_probe_tensor = (
        full_incoming_light_colors.cpu()
    )  # Copy back to CPU to get our probe tensor

    assert incoming_light_dirs is not None
    incoming_light_probe_tensor_directions = incoming_light_dirs[
        0
    ].cpu()  # Constant for every single point, so no need to keep track of all of them and waste space
    print("Generating Probe Query Tensor...")
    # Get how close we are to each of the other points
    # Compute nearest neighbor for the point clouds a batch at a time to save memory.
    probe_query_batch_size = cast(int, args.caching_batch_size)
    point_cloud_batches = torch.split(
        collapsed_point_cloud, probe_query_batch_size, dim=0
    )  # List of (batch_size, 3)

    min_distances = torch.empty(0,).cuda()
    closest_points = torch.empty(0,).cuda()

    P = probe_point_xyz.size(0)
    for i, pc_batch in tqdm(enumerate(point_cloud_batches), total=len(point_cloud_batches)):
        # pc_batch has shape (B, 3)
        B = pc_batch.size(0)
        # Expand the point cloud and probe points to be the same size: (B, P, 3), then reduce the last dimension and take topKs.
        expanded_pc_batch = pc_batch[:, None, :].expand(-1, P, -1) # (B, P, 3) - add a singleton dimension then expand
        expanded_probe_batch = probe_point_xyz[None, :, :].expand(B, -1, -1) # (B, P, 3)
        # Get the distances from each point to a probe point
        all_pair_distances = torch.norm(expanded_pc_batch - expanded_probe_batch, p=2, dim=-1) # (B, P)

        # Get the closest points via min
        values, indices = torch.min(all_pair_distances, dim=1)  # Both tensors (B,)

        # Build our result
        min_distances = torch.cat([min_distances, values])
        closest_points = torch.cat([closest_points, indices])

    print("Incoming Light Probe Statistics:")
    print(f"{torch.mean(min_distances) = }")
    print(f"{closest_points = }")

    # Our closest points tensor should now be reshaped back to its more-structured (N, H, W, 1) shape and then saved out
    closest_points = closest_points.view(num_cameras, global_image_height, global_image_width, 1)
    closest_points = closest_points.permute(0, 3, 1, 2) # (N, 1, H, W)

    incoming_light_probe_query_tensor = closest_points.cpu()

    # Save out all of our tensors into a dictionary.
    cache_save_dict: BRDFCacheDict = {
        "full_rendered_images": full_rendered_images_tensor,
        "incoming_light_probe_colors": incoming_light_probe_tensor,
        "incoming_light_probe_directions": incoming_light_probe_tensor_directions,
        "incoming_light_probe_query": incoming_light_probe_query_tensor,
    }

    print("Saving Tensors...")
    os.makedirs(cache_save_folder, exist_ok=True)
    print(f"Cache Dir: {cache_save_folder.absolute()}")
    torch.save(cache_save_dict, cache_save_folder / "full_cache_dict.pt")
    print(f"Saved full cache dictionary at {cache_save_folder / 'full_cache_dict.pt'}")
