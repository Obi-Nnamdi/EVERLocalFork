"""
Usage:
python visualize_light_cache.py -m /data/trained_model --cache_location=/data/trained_model/brdf_ever_cache/full_cache_dict.pt
"""

import torch
from arguments import (
    ModelParams,
    PipelineParams,
    OptimizationParams,
    BRDFOptmizationParams,
)
from argparse import ArgumentParser
from raytracing import (
    get_cameras,
    save_rgb_image,
    load_gaussian_model
)
from cache_incoming_light import BRDFCacheDict
import sys
from tqdm import tqdm

from utils.tensor_utils import nchw_tensor_to_npc

import torch
import math

from pathlib import Path
import os

from typing import cast, TypedDict

import pandas as pd

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Light Cache Visualization Parameters")
    parser.add_argument(
        "--start_ever_checkpoint",
        type=str,
        default=None,
        help="Checkpoint to resume ever model from.",
    )
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    brdf_optim_params = BRDFOptmizationParams(parser)
    args = parser.parse_args(sys.argv[1:])
    brdf_args = cast(
        BRDFOptmizationParams, brdf_optim_params.extract(args)
    )  # NOTE: Lying to the type checker, but it's close enough.
    model_params = cast(ModelParams, lp.extract(args))
    optim_params = cast(OptimizationParams, op.extract(args))

    print(f"Loading cache from {brdf_args.cache_location}...")
    cache_dict = cast(
        BRDFCacheDict, torch.load(Path(brdf_args.cache_location), map_location="cuda")
    )

    print("Performing Point Cloud Operations...")
    # Define Save dir:
    visualization_save_dir = Path(model_params._model_path) / "viz"
    os.makedirs(visualization_save_dir, exist_ok=True)

    downsample_stride = 50
    # Get a (N * D, 7) point cloud consisting of (X, Y, Z, R, G, B, (Index of Light Probe)) information
    downsampled_scene_xyz = cache_dict["full_scene_point_cloud"][:, ::downsample_stride] # (All cameras, with pointcloud downsampled)
    downsampled_colors = nchw_tensor_to_npc(cache_dict["full_rendered_images"])[:, ::downsample_stride, :3] # (just want R,G,B)
    downsampled_light_probe_idx = nchw_tensor_to_npc(cache_dict["light_probe_query"])[:, ::downsample_stride]

    downsampled_pc = torch.concat([downsampled_scene_xyz, downsampled_colors, downsampled_light_probe_idx], dim = -1) # (N, D, 7)

    # Collapse first two dimensions to get matrix (N * D, 6)
    downsampled_pc = downsampled_pc.reshape(-1, downsampled_pc.size(2))

    # Save out the point cloud as a CSV
    numpy_pc = downsampled_pc.cpu().numpy()
    pc_dataframe = pd.DataFrame(numpy_pc, columns=["x", "y", "z", "r", "g", "b", "probe_idx"])
    scene_pc_filename = visualization_save_dir / f"scene_downsampled_pc_{downsample_stride}.csv"
    pc_dataframe.to_csv(scene_pc_filename, index=False)
    print(f"Saved Scene Point Cloud CSV at {scene_pc_filename}")


    print("Performing Camera Operations...")
    cameras = get_cameras(model_params)
    # print(f"Camera World-To-Cam Transform: \n{cameras[1].world_view_transform}")
    # fovy doesn't matter.
    print(f"Blender Camera FOV: {math.degrees(cameras[1].FoVx)} deg")
    print(f"{cameras[1].image_width = }")
    print(f"{cameras[1].image_height = }")
    
    # Writing transforms to CSV in column-major order
    print(f"Saving Camera Transforms: ")
    # Getting column names to write to CSV
    mat_size = 4 # (4 x 4) Transformation Matrices
    col_indices, row_indices = torch.meshgrid(torch.arange(mat_size), torch.arange(mat_size), indexing="ij")
    camera_transform_column_names = [f"c{i + 1}_r{j + 1}" for i, j in zip(col_indices.ravel(), row_indices.ravel())]


    # Convert to dataframe then to CSV
    all_transforms = torch.stack([camera.world_view_transform.T.ravel() for camera in cameras], dim=0)
    all_transforms_numpy = all_transforms.cpu().numpy()
    all_transforms_df = pd.DataFrame(all_transforms_numpy, columns=camera_transform_column_names)

    camera_transform_csv_filename = visualization_save_dir / f"camera_transforms.csv"
    all_transforms_df.to_csv(camera_transform_csv_filename, index=False)

    # Get our incoming / outgoing light saved to CSV
    print(f"Visualizing Incoming / Outgoing Light:")

    # Create a (P, R * 3 + R * 3) point cloud of the incoming and outgoing light at a point
    P, R, _ = cache_dict["incoming_light_probe_colors"].shape
    flattened_incoming_colors = cache_dict["incoming_light_probe_colors"].reshape(P, -1) # (P, R * 3)
    flattened_outgoing_colors = cache_dict["outgoing_light_probe_colors"].reshape(P,  -1) # (P, R * 3)

    all_probe_colors = torch.cat([flattened_incoming_colors, flattened_outgoing_colors], dim=-1)

    # Define column names for dataframe
    incoming_color_cols: list[str] = []
    outgoing_color_cols: list[str] = []
    for i in range(R):
        incoming_light_col_names = [f"in_ray_{i}_r", f"in_ray_{i}_g", f"in_ray_{i}_b"]
        outgoing_light_col_names = [f"out_ray_{i}_r", f"out_ray_{i}_g", f"out_ray_{i}_b"]

        incoming_color_cols.extend(incoming_light_col_names)
        outgoing_color_cols.extend(outgoing_light_col_names)

    all_col_names = incoming_color_cols + outgoing_color_cols

    # Create dataframe and save out to CSV
    probe_colors_df = pd.DataFrame(all_probe_colors.cpu().numpy(), columns=all_col_names)
    light_probe_save_filename = visualization_save_dir / f"light_probe_colors.csv"
    probe_colors_df.to_csv(light_probe_save_filename)
    print(f"Saved light probe to {light_probe_save_filename}")

    # Save out our incoming light directions
    light_probe_directions_df = pd.DataFrame(cache_dict["light_probe_directions"].cpu().numpy(), columns=["x", "y", "z"])
    light_probe_direction_save_filename = visualization_save_dir / "light_probe_directions.csv"

    light_probe_directions_df.to_csv(light_probe_direction_save_filename)
    print(f"Saved light probe directions to {light_probe_direction_save_filename}")

    # Save all the images we've rendered out to disk for later blender visualization
    image_save_folder = visualization_save_dir / "rendered_images"
    os.makedirs(image_save_folder, exist_ok=True)
    print(f"Saving rendered images to disk at {image_save_folder.absolute()}")
    for index, image in enumerate(cache_dict["full_rendered_images"]):
        image_name = f"rendered_img_{index:03d}.jpg"
        save_rgb_image(image, image_save_folder / image_name)

    # Save out our original gaussians for visualization
    print("Loading Gaussians...")
    gaussians = load_gaussian_model(model_params, optim_params, args.start_ever_checkpoint)
    print(f"Loaded Gaussian, Active SH Degree: {gaussians.active_sh_degree}")
    gaussian_ply_file_name = visualization_save_dir / "gaussians.ply"
    gaussians.save_blender_ply(gaussian_ply_file_name)
    print(f"Saved Gaussians to disk at {gaussian_ply_file_name.absolute()}")
