from arguments import (
    ModelParams,
    PipelineParams,
    OptimizationParams,
    BRDFOptmizationParams,
)
from argparse import ArgumentParser
from neural_brdf import (
    BRDF_normal_predictor,
    transform_normals_to_world_space,
    eval_blinn_phong_outgoing_radiance,
    FullModelOutput,
)
from raytracing import (
    build_gaussian_renderer,
    depth_map_to_xyz,
    get_cameras,
    load_gaussian_model,
    render_gaussians,
    generate_spherical_rays,
    gather_incoming_light_at_points,
    plot_incoming_light_and_outgoing_radiance,
    plot_outgoing_radiance_for_multiple_cameras,
)
from utils.general_utils import safe_state
import sys
from tqdm import tqdm

import torch
from torch import nn

from pathlib import Path
import os

from typing import cast
from datetime import datetime

from torch.utils.tensorboard.writer import SummaryWriter
# Graphing
import matplotlib

matplotlib.use("Agg")  # headless mode
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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


def p_by_c_tensor_to_chw(input_tensor: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Converts a (P, C) tensor into a (C, H, W) tensor, assuming that P was created by collapsing the (H,W) dimensions.
    E.g. (H * W, 3) -> (1, 3, H, W)
    """
    P, C = input_tensor.shape
    input_tensor = input_tensor.reshape(H, W, C)  # (H, W, C)
    input_tensor = input_tensor.permute(2, 0, 1)  # (C, H, W)

    return input_tensor


def pretty_display_normal_tensor(normal_tensor: torch.Tensor) -> torch.Tensor:
    """
    Cleans up a normal tensor by mapping it from (-1, 1) -> (0, 1)
    """
    return (normal_tensor / 2) + 0.5


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
    args = parser.parse_args(sys.argv[1:])
    brdf_args = cast(
        BRDFOptmizationParams, brdf_optim_params.extract(args)
    )  # NOTE: Lying to the type checker, but it's close enough.

    print("Optimizing " + args.model_path)

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

    # Create our SummaryWriter for training logging
    model_checkpoint_dir = "brdf_models"
    model_save_path = (
        Path(model_params.model_path) / model_checkpoint_dir / "brdf_model.pt"
    )
    os.makedirs(model_save_path.parent, exist_ok=True)

    # Create filename-safe current date (ideally should be in system/local time)
    curr_date_str = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=model_save_path.parent / "runs" / curr_date_str)
    print(f"Model Running Directory: {model_save_path.parent.absolute()}")

    # Set a global image width and height that is used for instanciating the neural network, etc.
    global_image_height = cast(
        int, rendering_cameras[0].image_height // brdf_args.preview_factor
    )
    global_image_width = cast(
        int, rendering_cameras[0].image_width // brdf_args.preview_factor
    )

    print(
        f"Rendering images at a {global_image_width} x {global_image_height} resolution (W x H)."
    )

    if brdf_args.randomly_sample_loss:
        print(
            f"Sampling {brdf_args.point_batch_size} pixels per iteration for loss calculation."
        )

    # Set up our initial renderer
    ever_renderer = build_gaussian_renderer(
        gaussians, rendering_cameras[0], cast(PipelineParams, pp.extract(args))
    )

    # More constants (affecting how much incoming light we use)
    incoming_light_sphere_divisions = 20

    # Calculate how big our incoming light features will be when input into our model.
    # test_sphere_o, _ = generate_spherical_rays(
    #     torch.zeros((3,)), incoming_light_sphere_divisions
    # )
    # incoming_light_size = test_sphere_o.size(0) * 3  # N vectors that have [r, g, b]

    # Instanciate the BRDF_normal_predictor
    brdf_normal_model = BRDF_normal_predictor(global_image_height, global_image_width)
    brdf_normal_model = brdf_normal_model.cuda()
    brdf_normal_model.train()

    # Load checkpoint if we're resuming from a checkpoint
    if brdf_args.resume_from != "":
        print(f"Loading checkpoint from {brdf_args.resume_from}.")
        model_state_dict = torch.load(Path(brdf_args.resume_from))
        brdf_normal_model.load_state_dict(model_state_dict)
        print(f"Loaded model checkpoint.")

    # Training Config
    loss_fn = nn.MSELoss()
    color_penalty = 0.5

    lr = 0.001
    grad_norm_clip = 1
    optimizer = torch.optim.AdamW(
        brdf_normal_model.parameters(),
        lr=lr,
    )

    # TODO: Learning rate scheduler
    # TODO: etc, etc.

    # reserved_mem = torch.cuda.memory.memory_reserved() / 1024 / 1024 / 1024
    # allocated_mem = torch.cuda.memory.memory_allocated() / 1024 / 1024 / 1024
    # print(f"{reserved_mem = }")
    # Main training loop:
    for step_num in tqdm(range(brdf_args.training_steps)):
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        # Random Camera index and chosen point
        camera_index = int(torch.randint(num_cameras, (1,)).item())

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
        model_output = cast(
            FullModelOutput, brdf_normal_model(rendered_image.unsqueeze(0))
        )
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
        rand_points = None
        if brdf_args.randomly_sample_loss:
            # uniformly choose a few points at a time to calculate loss with on low VRAM configs.
            multinom_weights = torch.ones(
                (global_image_height * global_image_width,)
            ).cuda()
            rand_points = torch.multinomial(
                multinom_weights, brdf_args.point_batch_size, replacement=False
            )  # (P, )

            # TODO: Replace with this code? (would have to time it)
            # rand_points = torch.randperm(
            #     global_image_height * global_image_width, device="cuda"
            # )[: brdf_args.point_batch_size]

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

        # TODO: How to handle "color penalty"? Maybe penalize each of the maps from getting too far away from the main rendered color idk.
        # loss = (
        #     loss_fn(outgoing_radiance, rendered_colors)
        #     + color_penalty * torch.norm(Ks)
        #     + torch.norm(Kd)
        # )
        loss = loss_fn(outgoing_radiance, rendered_colors)
        loss.backward()
        # Clip grad norms
        # torch.nn.utils.clip_grad_norm_(brdf_normal_model.parameters(), grad_norm_clip)

        # Check gradient norms
        parameters = brdf_normal_model.conv1.parameters()
        norm_type = 2
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )

        optimizer.step()

        # Add loss and other metrics
        writer.add_scalar("Train/loss", loss.detach().cpu(), step_num)
        writer.add_scalar("Train/grad_norm_1", total_norm.cpu(), step_num)

        if step_num % brdf_args.image_reporting_interval == 0:
            # Report loss and other metrics
            tqdm.write(f"========Step {step_num}:========")
            tqdm.write(f"Camera Index: {camera_index}")
            tqdm.write(f"{outgoing_radiance = }")
            tqdm.write(f"{loss = }")

            # Add images for diffuse, specular, original, etc.
            # TODO: Could be a matplotlib figure
            # TODO: Add normal and spec_c images?

            writer.add_image(
                f"camera_image", rgb_image.clip(0, 1), step_num, dataformats="HWC"
            )
            writer.add_image(
                f"diffuse_output",
                model_output["brdf"]["diffuse"][0].clip(0, 1),
                step_num,
            )
            writer.add_image(
                f"specular_output",
                model_output["brdf"]["specular"][0].clip(0, 1),
                step_num,
            )
            writer.add_image(
                f"specular_c_mapped_0_1",
                model_output["brdf"]["specular_c"][0] / brdf_normal_model.max_spec_c,
                step_num,
            )

            if brdf_args.randomly_sample_loss:
                # Recalculate world normals since they may be subsampled.
                # TODO: Handle this better, as it takes additional VRAM.
                full_cam_normals = nchw_tensor_to_p_by_c(
                    model_output["normal"]
                )  # (X, 3)
                full_cam_normals = nn.functional.normalize(
                    full_cam_normals, dim=1
                )  # (X, 3)
                full_world_normals = transform_normals_to_world_space(
                    full_cam_normals, rendering_cam
                )  # (X, 3)
            else:
                full_cam_normals = camera_normals_normed
                full_world_normals = world_normals

            writer.add_image(
                f"camera_normal",
                pretty_display_normal_tensor(
                    p_by_c_tensor_to_chw(
                        full_cam_normals, global_image_height, global_image_width
                    )
                ),
                step_num,
            )
            writer.add_image(
                f"world_normal",
                pretty_display_normal_tensor(
                    p_by_c_tensor_to_chw(
                        full_world_normals, global_image_height, global_image_width
                    )
                ),
                step_num,
            )

            # Plot incoming light at a random point for visualization
            if not brdf_args.randomly_sample_loss:
                # Choose a random point and plot the incoming -> outgoing radiance
                rand_row = int(torch.randint(global_image_height, (1,)).item())
                rand_col = int(torch.randint(global_image_width, (1,)).item())
                point_loc = (rand_row, rand_col)

                point_index = (
                    point_loc[0] * global_image_width + point_loc[1]
                )  # Row-major order`
            else:
                # Use an index of the "rand_points" array instead (we've downsampled our output)
                assert rand_points is not None
                point_index = int(torch.randint(rand_points.size(0), (1,)).item())

                # Recalculate the true location in the image it's from.
                actual_point = int(rand_points[point_index].item())
                img_row = actual_point // global_image_width
                img_col = actual_point % global_image_width
                point_loc = (img_row, img_col)

            fig = plt.figure(dpi=300, figsize=(9, 3))

            ax = fig.add_subplot(1, 2, 1, projection="3d")
            plt.suptitle(
                f"Incoming Light for Camera {camera_index} at point {point_loc} (row, col)"
            )

            plot_incoming_light_and_outgoing_radiance(
                incoming_light_colors[point_index],
                incoming_light_dirs[point_index],
                outgoing_radiance[point_index : point_index + 1],
                outgoing_directions[point_index : point_index + 1],
            )

            # Show the point we used to generate the plot
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(rgb_image.cpu().clip(0, 1))
            ax.plot(point_loc[1], point_loc[0], marker="x", color="r")
            # ax.set_xlim(point_loc[0] - 200, point_loc[0] + 200)
            # ax.set_aspect("equal")

            # plt.savefig(model_save_path.parent.parent / "incoming_light_test.png")
            writer.add_figure("incoming_light_readout", fig, step_num)

            ##################### OUTGOING RADIANCE PLOT #####################
            # Plot outgoing BRDF for a single point by choosing a single BRDF value and varying camera angles
            kd_value = Kd[point_index : point_index + 1]  # (1, 3)
            ks_value = Ks[point_index : point_index + 1]  # (1, 3)
            spec_c_value = spec_c[point_index : point_index + 1]  # (1,)
            normal_value = world_normals[point_index : point_index + 1]  # (1, 3)
            incoming_light_color_val = incoming_light_colors[
                point_index : point_index + 1
            ]  # (1, N, 3)
            incoming_light_dir_val = incoming_light_dirs[
                point_index : point_index + 1
            ]  # (1, N, 3)

            # Generate the rays we're going to query our BRDF with (spherical rays)
            point_outgoing_dirs, _ = generate_spherical_rays(
                torch.tensor([0.0, 0.0, 0.0]), incoming_light_sphere_divisions
            )
            point_outgoing_dirs = point_outgoing_dirs.cuda()  # (N, 3)

            # Generate the rays we're going to query our BRDF with (camera rays)
            camera_pos = rendering_cam.camera_center.cuda()  # (3,)
            outgoing_directions = nn.functional.normalize(
                (camera_pos - all_points_xyz), dim=1
            )  # (P, 3)

            all_camera_origins = torch.stack(
                [cam.camera_center for cam in rendering_cameras], dim=0
            ).cuda()  # (M, 3)
            all_camera_dirs = all_camera_origins - all_points_xyz[point_index]  # (M, 3)
            all_camera_dirs = nn.functional.normalize(all_camera_dirs, dim=1)  # (M, 3)

            M = point_outgoing_dirs.size(
                0
            )  # How many rays are we querying BRDF for? This becomes the outer "P" dimension.
            outgoing_radiance_colors = eval_blinn_phong_outgoing_radiance(
                incoming_light_color_val.expand(M, -1, -1).contiguous(),
                incoming_light_dir_val.expand(M, -1, -1).contiguous(),
                point_outgoing_dirs,
                normal_value.expand(M, -1).contiguous(),
                kd_value.expand(M, -1).contiguous(),
                ks_value.expand(M, -1).contiguous(),
                spec_c_value.expand(
                    M,
                ).contiguous(),
            )

            # Plot the result
            original_outgoing_radiance = outgoing_radiance[
                point_index : point_index + 1
            ]  # (1, 3)
            original_outgoing_direction = outgoing_directions[
                point_index : point_index + 1
            ]  # (1, 3)

            fig = plt.figure(dpi=300, figsize=(9, 3))
            ax = fig.add_subplot(1, 2, 1, projection="3d")
            plt.suptitle(
                f"Outgoing Radiance for Camera {camera_index} at point {point_loc} (row, col)"
            )
            plot_outgoing_radiance_for_multiple_cameras(
                outgoing_radiance_colors,
                point_outgoing_dirs,
                original_outgoing_radiance,
                original_outgoing_direction,
            )
            # Show the point we used to generate the plot
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(rgb_image.cpu().clip(0, 1))
            ax.plot(point_loc[1], point_loc[0], marker="x", color="r")

            # TODO: Show other camera perspectives for comparison (i.e. +-1 camera indices)?

            # plt.savefig(model_save_path.parent.parent / "outgoing_light_test.png")
            writer.add_figure("outgoing_light_readout", fig, step_num)

            writer.flush()

        if step_num % brdf_args.checkpoint_interval == 0 and step_num != 0:
            model_checkpoint_path = model_save_path.parent / f"brdf_model_{step_num}.pt"
            torch.save(
                brdf_normal_model.state_dict(),
                model_checkpoint_path,
            )
            tqdm.write(f"Saved model at {model_checkpoint_path.absolute()}")
    # Flush writer
    writer.flush()
    # TODO: Save model, optimizer, and epoch for resuming
    torch.save(brdf_normal_model.state_dict(), model_save_path)
    print(f"Saved model at {model_save_path.absolute()}")
    # All done
