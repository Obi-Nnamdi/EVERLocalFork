from arguments import (
    ModelParams,
    PipelineParams,
    OptimizationParams,
    BRDFOptmizationParams,
)
from argparse import ArgumentParser
from neural_brdf import (
    BRDF_normal_predictor,
    batch_transform_normals_to_world_space,
    eval_blinn_phong_outgoing_radiance,
    FullModelOutput,
)

from batch_eval_blinn_phong_brdf import (
    batch_eval_blinn_phong_outgoing_radiance_with_probe,
    calc_optimal_batch_size_for_brdf_eval,
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
from cache_incoming_light import BRDFCacheDict
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

from utils.tensor_utils import (
    nchw_tensor_to_npc,
    npc_tensor_to_nchw,
    pretty_display_normal_tensor,
)

matplotlib.use("Agg")  # headless mode
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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
    model_params: ModelParams = cast(ModelParams, lp.extract(args))
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    rendering_cameras = get_cameras(model_params)
    # Create our SummaryWriter for training logging
    model_checkpoint_dir = "brdf_models"
    model_save_path = (
        Path(model_params._model_path) / model_checkpoint_dir / "brdf_model.pt"
    )
    os.makedirs(model_save_path.parent, exist_ok=True)

    # Create filename-safe current date (ideally should be in system/local time)
    curr_date_str = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=model_save_path.parent / "runs" / curr_date_str)
    print(f"Model Running Directory: {model_save_path.parent.absolute()}")

    # Load cache dir and tensors:
    assert brdf_args.cache_location != ""
    print(f"Loading cache from {brdf_args.cache_location}.")
    cache_dict = cast(
        BRDFCacheDict, torch.load(Path(brdf_args.cache_location), map_location="cuda")
    )

    # Extract all tensors
    rendered_images = cache_dict["full_rendered_images"]  # (N, 4, H, W)
    full_scene_point_cloud = cache_dict["full_scene_point_cloud"]  # (N, H * W, 3)
    probe_incoming_light_colors = cache_dict["incoming_light_probe_colors"]  # (P, R, 3)
    probe_incoming_light_directions = cache_dict[
        "incoming_light_probe_directions"
    ]  # (R, 3)
    incoming_light_query_mapping = cache_dict[
        "incoming_light_probe_query"
    ]  # (N, 1, H, W)

    # Reshape query probe to be ready for putting into the slangtorch kernel
    incoming_light_query_mapping = (
        nchw_tensor_to_npc(incoming_light_query_mapping).squeeze(-1).contiguous()
    )  # (N, HW)

    # Use cache to get global image height and width:
    global_image_height = rendered_images.size(2)
    global_image_width = rendered_images.size(3)

    print(
        f"Loaded Images are at {global_image_width} x {global_image_height} (w x h) resolution."
    )

    # Handle creation of some early tensor operations we'll always use throughout training
    camera_positions = torch.stack(
        [camera.camera_center.cuda() for camera in rendering_cameras], dim=0
    )  # (N, 3)

    outgoing_directions = nn.functional.normalize(
        (
            camera_positions[:, None, :].expand(
                -1, global_image_height * global_image_width, -1
            )
            - full_scene_point_cloud
        ),
        dim=-1,
    )  # (N, H * W, 3)

    # Extract the RGB colors for each point from the rendered images
    rendered_colors = nchw_tensor_to_npc(rendered_images[:, :3])  # (N, HW, 3)

    # Min-max normalize the depth in log space
    rendered_images[:, 3] = torch.log(rendered_images[:, 3])
    assert not torch.any(rendered_images.isinf())  # Ensure we have no bad values

    max_depth_val = torch.max(rendered_images[:, 3])
    min_depth_val = torch.min(rendered_images[:, 3])

    # [0 - 1 Normalization]
    rendered_images[:, 3] -= min_depth_val
    rendered_images[:, 3] /= max_depth_val - min_depth_val

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

        # Ask for our BRDF values
        model_output = cast(FullModelOutput, brdf_normal_model(rendered_images))
        # print(f"{model_output = }")

        # Collect Model Outputs
        Kd = nchw_tensor_to_npc(model_output["brdf"]["diffuse"])  # (N, H * W, 3)
        Ks = nchw_tensor_to_npc(model_output["brdf"]["specular"])  # (N, H * W, 3)
        spec_c = nchw_tensor_to_npc(model_output["brdf"]["specular_c"]).squeeze(
            -1
        )  # (N, H * W, )
        camera_normals_unnormed = nchw_tensor_to_npc(
            model_output["normal"]
        )  # (N, H * W, 3)

        # BRDF reconstruction
        camera_normals_normed = nn.functional.normalize(camera_normals_unnormed, dim=1)
        world_normals = batch_transform_normals_to_world_space(
            camera_normals_normed, rendering_cameras
        )  # (N, H * W, 3)

        outgoing_radiance = batch_eval_blinn_phong_outgoing_radiance_with_probe(
            probe_incoming_light_colors,
            probe_incoming_light_directions,
            incoming_light_query_mapping,
            outgoing_directions,
            world_normals,
            Ks,
            Kd,
            spec_c,
            None,  # TODO: Determine a constant sub_batch_size (12 is around okay)
        )  # (B, HW, 3)

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
            torch.stack(
                [
                    torch.norm(p.grad.detach(), norm_type) for p in parameters
                ]  # pyright: ignore[reportOptionalMemberAccess]
            ),
            norm_type,
        )

        optimizer.step()

        # Add loss and other metrics
        writer.add_scalar("Train/loss", loss.detach().cpu(), step_num)
        writer.add_scalar("Train/grad_norm_1", total_norm.cpu(), step_num)

        if step_num % brdf_args.image_reporting_interval == 0:
            image_reporting_batch_size = 4  # (batch dimension for showing our images)
            # Report loss and other metrics
            tqdm.write(f"========Step {step_num}:========")
            tqdm.write(f"{outgoing_radiance = }")
            tqdm.write(f"{loss = }")
            # TODO: Write down incoming light values?

            writer.add_images(
                f"camera_rgb_images",
                rendered_images[:image_reporting_batch_size, :3].clip(0, 1),
                step_num,
                dataformats="NCHW",
            )
            writer.add_images(
                f"camera_depth_images",
                rendered_images[:image_reporting_batch_size, 3:4].clip(0, 1),
                step_num,
                dataformats="NCHW",
            )
            """
            Show the reconstructed images
            """
            writer.add_images(
                f"reconstructed_camera_images",
                npc_tensor_to_nchw(
                    outgoing_radiance[:image_reporting_batch_size],
                    global_image_height,
                    global_image_width,
                ).clip(0, 1),
                step_num,
            )

            writer.add_images(
                f"diffuse_output",
                model_output["brdf"]["diffuse"][:image_reporting_batch_size].clip(0, 1),
                step_num,
            )
            writer.add_images(
                f"specular_output",
                model_output["brdf"]["specular"][:image_reporting_batch_size].clip(
                    0, 1
                ),
                step_num,
            )
            writer.add_images(
                f"specular_c_mapped_0_1",
                model_output["brdf"]["specular_c"][:image_reporting_batch_size]
                / brdf_normal_model.max_spec_c,
                step_num,
            )

            writer.add_images(
                f"camera_normal",
                pretty_display_normal_tensor(
                    npc_tensor_to_nchw(
                        camera_normals_normed[:image_reporting_batch_size], global_image_height, global_image_width
                    )
                ),
                step_num,
            )
            writer.add_images(
                f"world_normal",
                pretty_display_normal_tensor(
                    npc_tensor_to_nchw(
                        world_normals[:image_reporting_batch_size], global_image_height, global_image_width
                    )
                ),
                step_num,
            )

            # Plot incoming light at a random point for visualization
            # Choose a random point and plot the incoming -> outgoing radiance
            rand_camera_index = int(
                torch.randint(outgoing_radiance.size(0), (1,)).item()
            )
            rand_row = int(torch.randint(global_image_height, (1,)).item())
            rand_col = int(torch.randint(global_image_width, (1,)).item())
            point_loc = (rand_row, rand_col)

            point_index = (
                point_loc[0] * global_image_width + point_loc[1]
            )  # Row-major order`

            fig = plt.figure(dpi=300, figsize=(9, 3))

            ax = fig.add_subplot(1, 2, 1, projection="3d")
            plt.suptitle(
                f"Incoming Light for Camera {rand_camera_index} at point {point_loc} (row, col)"
            )

            # Get the point's associated incoming light
            incoming_light_probe_point_index = incoming_light_query_mapping[
                rand_camera_index, point_index
            ]
            point_incoming_light_color = probe_incoming_light_colors[
                incoming_light_probe_point_index
            ]
            cam_matplotlib_image = rendered_images[rand_camera_index, :3].permute(
                1, 2, 0
            )  # (H, W, 3)

            plot_incoming_light_and_outgoing_radiance(
                point_incoming_light_color,
                probe_incoming_light_directions,
                outgoing_radiance[rand_camera_index, point_index : point_index + 1],
                outgoing_directions[rand_camera_index, point_index : point_index + 1],
            )

            # Show the point we used to generate the plot
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(cam_matplotlib_image.cpu().clip(0, 1))
            ax.plot(point_loc[1], point_loc[0], marker="x", color="r")
            # ax.set_xlim(point_loc[0] - 200, point_loc[0] + 200)
            # ax.set_aspect("equal")

            writer.add_figure("incoming_light_readout", fig, step_num)

            ##################### OUTGOING RADIANCE PLOT #####################
            # Plot outgoing BRDF for a single point by choosing a single BRDF value and varying camera angles
            kd_value = Kd[rand_camera_index, point_index : point_index + 1]  # (1, 3)
            ks_value = Ks[rand_camera_index, point_index : point_index + 1]  # (1, 3)
            spec_c_value = spec_c[
                rand_camera_index, point_index : point_index + 1
            ]  # (1,)
            normal_value = world_normals[
                rand_camera_index, point_index : point_index + 1
            ]  # (1, 3)
            incoming_light_color_val = point_incoming_light_color[None, :]  # (1, N, 3)

            # Generate the rays we're going to query our BRDF with (spherical rays)
            point_outgoing_dirs, _ = generate_spherical_rays(
                torch.tensor([0.0, 0.0, 0.0]), brdf_args.incoming_light_divisions
            )
            point_outgoing_dirs = point_outgoing_dirs.cuda()  # (N, 3)

            M = point_outgoing_dirs.size(
                0
            )  # How many rays are we querying BRDF for? This becomes the outer "P" dimension.
            outgoing_radiance_colors = eval_blinn_phong_outgoing_radiance(
                incoming_light_color_val.expand(M, -1, -1).contiguous(),
                probe_incoming_light_directions,
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
                rand_camera_index, point_index : point_index + 1
            ]  # (1, 3)
            original_outgoing_direction = outgoing_directions[
                rand_camera_index, point_index : point_index + 1
            ]  # (1, 3)

            fig = plt.figure(dpi=300, figsize=(9, 3))
            ax = fig.add_subplot(1, 2, 1, projection="3d")
            plt.suptitle(
                f"Outgoing Radiance for Camera {rand_camera_index} at point {point_loc} (row, col)"
            )
            plot_outgoing_radiance_for_multiple_cameras(
                outgoing_radiance_colors,
                point_outgoing_dirs,
                original_outgoing_radiance,
                original_outgoing_direction,
            )
            # Show the point we used to generate the plot
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(cam_matplotlib_image.cpu().clip(0, 1))
            ax.plot(point_loc[1], point_loc[0], marker="x", color="r")

            # TODO: Show other camera perspectives for comparison (i.e. +-1 camera indices)?
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
