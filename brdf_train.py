from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser
from neural_brdf import (
    BRDF_normal_predictor,
    transform_normals_to_world_space,
    eval_blinn_phong_outgoing_radiance,
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


def nchw_tensor_to_p_by_c(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a (N, C, H, W) tensor into a (P, C) tensor, assuming that N (first dim) is 1, and there a P = H * W points.
    E.g. (1, 3, H, W) -> (H * W, 3)
    E.g. (1, 1, H, W) -> (H * W, 1)
    """
    N, C, H, W = input_tensor.shape
    input_tensor = input_tensor.squeeze(0)  # (C, H, W)
    input_tensor = input_tensor.reshape(C, H * W)
    input_tensor = input_tensor.T  # (H * W, C)

    return input_tensor


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

    rendering_cameras = get_cameras(model_params)
    num_cameras = len(rendering_cameras)

    # Set a global image width and height that is used for instanciating the neural network, etc.
    preview_factor = 16
    global_image_height = rendering_cameras[0].image_height // preview_factor
    global_image_width = rendering_cameras[0].image_width // preview_factor

    # Set up our initial renderer
    ever_renderer = build_gaussian_renderer(
        gaussians, rendering_cameras[0], pp.extract(args)
    )

    # More constants (affecting how much incoming light we use)
    incoming_light_sphere_divisions = 20

    # Calculate how big our incoming light features will be when input into our model.
    test_sphere_o, _ = generate_spherical_rays(
        torch.zeros((3,)), incoming_light_sphere_divisions
    )
    incoming_light_size = test_sphere_o.size(0) * 3  # N vectors that have [r, g, b]

    # Instanciate the BRDF_normal_predictor
    brdf_normal_model = BRDF_normal_predictor(global_image_height, global_image_width)
    brdf_normal_model = brdf_normal_model.cuda()
    brdf_normal_model.train()

    # Training Config
    loss_fn = nn.MSELoss()
    color_penalty = 0.5

    lr = 0.001
    grad_norm_clip = 1
    optimizer = torch.optim.AdamW(
        brdf_normal_model.parameters(),
        lr=lr,
    )

    randomly_sample_output = True  # Loss calculated only at randomly sampled points (specified by point_batch_size) to avoid high VRAM costs.

    training_steps = 500
    point_batch_size = 2048 * 6
    # TODO: Learning rate scheduler
    # TODO: etc, etc.

    # reserved_mem = torch.cuda.memory.memory_reserved() / 1024 / 1024 / 1024
    # allocated_mem = torch.cuda.memory.memory_allocated() / 1024 / 1024 / 1024
    # print(f"{reserved_mem = }")
    # Main training loop:
    for i in tqdm(range(training_steps)):
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        # TODO: Have random Camera index and chosen point
        camera_index = 1

        # Set up the camera we're rendering with
        # TODO: Can just render all images from all cameras once, right?
        rendering_cam = rendering_cameras[camera_index]
        # print(f"{global_image_width = }")
        # print(f"{global_image_height = }")
        rendering_cam.image_width = global_image_width
        rendering_cam.image_height = global_image_height

        rendered_image = render_gaussians(
            ever_renderer, rendering_cam, None, include_depth=True
        )  # (C, H, W)
        # TODO: Save the image output for debug purposes?
        # print("Rendered Image.")

        # Separate RGB and Depth Images
        rgb_image = rendered_image[:3, :, :].permute(1, 2, 0)  # (H, W, C)
        rendered_colors = rgb_image.reshape(-1, 3)  # (P, 3)

        depth_map = rendered_image[3, :, :]  # (H, W)

        # Make our xyz_map for each pixel
        rays_o, rays_d = ever_renderer.get_rays(rendering_cam)
        xyz_map = depth_map_to_xyz(rays_o, rays_d, depth_map)  # (H, W, 3)
        all_points_xyz = xyz_map.reshape(-1, 3)  # (H * W, 3)

        # Ask for our BRDF values
        model_output = brdf_normal_model(rendered_image.unsqueeze(0))
        # print(f"{model_output = }")

        # Collect Model Outputs
        Kd = nchw_tensor_to_p_by_c(model_output["brdf"]["diffuse"])  # (P, 3)
        Ks = nchw_tensor_to_p_by_c(model_output["brdf"]["specular"])  # (P, 3)
        spec_c = nchw_tensor_to_p_by_c(model_output["brdf"]["specular_c"]).squeeze(
            1
        )  # (P, )
        camera_normals_unnormed = nchw_tensor_to_p_by_c(
            model_output["normal"]
        )  # (P, 3)

        # Select only random points if we're sampling:
        if randomly_sample_output:
            # uniformly choose a few points at a time to calculate loss with on low VRAM configs.
            multinom_weights = torch.ones(
                (global_image_height * global_image_width,)
            ).cuda()
            rand_points = torch.multinomial(
                multinom_weights, point_batch_size, replacement=False
            )  # (P, )

            # Select only those indices for all relevant tensors
            all_points_xyz = all_points_xyz[rand_points, :]
            rendered_colors = rendered_colors[rand_points, :]

            Kd = Kd[rand_points, :]
            Ks = Ks[rand_points, :]
            spec_c = spec_c[rand_points]
            camera_normals_unnormed = camera_normals_unnormed[rand_points, :]

        # Debug w/ fake values
        # Kd = torch.full_like(Kd, 0.2)
        # Ks = torch.full_like(Ks, 0.2)
        # spec_c = torch.full_like(spec_c, 2.0)
        # camera_normals_unnormed = torch.full_like(camera_normals_unnormed, 0.2)

        # print(f"{Kd.shape  = }")
        # print(f"{Ks.shape  = }")
        # print(f"{spec_c.shape  = }")
        # print(f"{camera_normals_unnormed.shape  = }")

        camera_normals_normed = nn.functional.normalize(camera_normals_unnormed, dim=1)
        world_normals = transform_normals_to_world_space(
            camera_normals_normed, rendering_cam
        )

        # Querying Spherical Directions
        incoming_light_tmin = 0.01
        incoming_light_colors, _, incoming_light_dirs = gather_incoming_light_at_points(
            all_points_xyz,
            ever_renderer,
            tmin=incoming_light_tmin,
            sphere_divisions=incoming_light_sphere_divisions,
            # TODO: Experiment with changing these parameters if I get bad incoming light values.
            fast=True,
            precompute_sh=False,
        )  # (P, N, 3)

        # print(f"{incoming_light_dirs = }")
        # print(f"{incoming_light_colors = }")
        # print(f"{incoming_light_colors.shape = }")
        # print(f"{incoming_light_dirs.shape = }")

        # BRDF reconstruction
        camera_pos = rendering_cam.camera_center.cuda()  # (3,)
        outgoing_directions = nn.functional.normalize(
            (camera_pos - all_points_xyz), dim=1
        )  # (P, 3)

        outgoing_radiance = eval_blinn_phong_outgoing_radiance(
            incoming_light_colors,
            incoming_light_dirs,
            outgoing_directions,
            world_normals,
            Kd,
            Ks,
            spec_c,
        )  # (P, 3)

        # print(f"{outgoing_radiance.shape = }")
        print(f"{outgoing_radiance = }")

        # TODO: How to handle "color penalty"? Maybe penalize each of the maps from getting too far away from the main rendered color idk.
        # loss = (
        #     loss_fn(outgoing_radiance, rendered_colors)
        #     + color_penalty * torch.norm(Ks)
        #     + torch.norm(Kd)
        # )
        loss = loss_fn(outgoing_radiance, rendered_colors)

        print(f"{loss = }")
        loss.backward()

        # Clip grad norms
        # torch.nn.utils.clip_grad_norm_(brdf_normal_model.parameters(), grad_norm_clip)

        # Check gradient norms
        # parameters = brdf_normal_model.fc1.parameters()
        # norm_type = 2
        # total_norm = torch.norm(
        #     torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)

        # print(f"{total_norm = }")

        optimizer.step()

    # TODO: Save model, optimizer, and epoch for resuming
    model_checkpoint_dir = "brdf_models"
    model_save_path = (
        Path(model_params.model_path) / model_checkpoint_dir / "brdf_model.pt"
    )
    os.makedirs(model_save_path.parent, exist_ok=True)
    torch.save(brdf_normal_model.state_dict(), model_save_path)
    print(f"Saved model at {model_save_path.absolute()}")
    # All done
